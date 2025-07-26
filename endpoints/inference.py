import os
import tempfile
from fastapi import APIRouter, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from google.cloud import storage
from peft import PeftModel

router = APIRouter()

MODEL_DIR = "/app/model"
GCS_BUCKET = "llm-garage-models"
GCS_PREFIX = "gemma-peft-vertex-output/model"

def download_model_from_gcs(bucket_name, gcs_prefix_for_model, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if os.path.exists(local_dir):
        for item in os.listdir(local_dir):
            item_path = os.path.join(local_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
    else:
        os.makedirs(local_dir, exist_ok=True)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix_for_model))
    if not blobs:
        raise FileNotFoundError(f"No files found in GCS at gs://{bucket_name}/{gcs_prefix_for_model}")
    downloaded_files = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_blob_path = blob.name[len(gcs_prefix_for_model):]
        if relative_blob_path.startswith("/"):
            relative_blob_path = relative_blob_path[1:]
        file_path = os.path.join(local_dir, relative_blob_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
        downloaded_files.append(relative_blob_path)
        print(f"🔍 [Download Debug] Downloaded: {relative_blob_path}")
    
    print(f"🔍 [Download Debug] Total files downloaded: {len(downloaded_files)}")
    print(f"🔍 [Download Debug] Files: {downloaded_files}")
    print(f"🔍 [Download Debug] Local directory contents:")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            print(f"🔍 [Download Debug]   {os.path.join(root, file)}")
    
    print(f"Downloaded model files from gs://{bucket_name}/{gcs_prefix_for_model} to {local_dir}")

@router.post("/inference")
async def hf_inference(request: Request):
    body = await request.json()
    print(f"🔍 [Backend Debug] Received request body: {body}")
    print(f"🔍 [Backend Debug] Body type: {type(body)}")
    print(f"🔍 [Backend Debug] Body keys: {list(body.keys()) if isinstance(body, dict) else 'Not a dict'}")
    
    request_id = body.get("request_id")  # Get request_id instead of model_name
    prompt = body.get("prompt")
    max_new_tokens = body.get("max_new_tokens", 100)
    base_model = body.get("base_model", "google/gemma-3-1b-pt")  # Default base model
    
    print(f"🔍 [Backend Debug] Extracted values:")
    print(f"🔍 [Backend Debug] request_id: {request_id} (type: {type(request_id)})")
    print(f"🔍 [Backend Debug] prompt: {prompt} (type: {type(prompt)})")
    print(f"🔍 [Backend Debug] base_model: {base_model}")
    print(f"🔍 [Backend Debug] max_new_tokens: {max_new_tokens}")
    
    if not request_id or not prompt:
        print(f"🔍 [Backend Debug] Validation failed: request_id={request_id}, prompt={prompt}")
        return {"error": "request_id and prompt are required"}
    
    try:
        # Download LoRA adapters from GCS
        adapter_gcs_path = f"{GCS_PREFIX}/{request_id}/final_model/"
        print(f"🔍 [Inference Debug] GCS path: gs://{GCS_BUCKET}/{adapter_gcs_path}")
        print(f"🔍 [Inference Debug] Local directory: {MODEL_DIR}")
        
        try:
            # Download to a temporary directory first, then move files to MODEL_DIR
            with tempfile.TemporaryDirectory() as temp_dir:
                download_model_from_gcs(GCS_BUCKET, adapter_gcs_path, temp_dir)
                
                # Move files from temp_dir/final_model/ to MODEL_DIR
                final_model_temp_dir = os.path.join(temp_dir, "final_model")
                if os.path.exists(final_model_temp_dir):
                    # Move all files from final_model subdirectory to MODEL_DIR
                    for item in os.listdir(final_model_temp_dir):
                        src = os.path.join(final_model_temp_dir, item)
                        dst = os.path.join(MODEL_DIR, item)
                        if os.path.isfile(src):
                            import shutil
                            shutil.copy2(src, dst)
                            print(f"🔍 [Inference Debug] Moved file: {item}")
                        elif os.path.isdir(src):
                            import shutil
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            print(f"🔍 [Inference Debug] Moved directory: {item}")
                else:
                    # If no final_model subdirectory, move files directly
                    for item in os.listdir(temp_dir):
                        src = os.path.join(temp_dir, item)
                        dst = os.path.join(MODEL_DIR, item)
                        if os.path.isfile(src):
                            import shutil
                            shutil.copy2(src, dst)
                            print(f"🔍 [Inference Debug] Moved file: {item}")
                        elif os.path.isdir(src):
                            import shutil
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            print(f"🔍 [Inference Debug] Moved directory: {item}")
                
                print(f"🔍 [Inference Debug] Final MODEL_DIR contents:")
                for item in os.listdir(MODEL_DIR):
                    print(f"🔍 [Inference Debug]   {item}")
                    
        except FileNotFoundError as e:
            print(f"🔍 [Inference Debug] FileNotFoundError: {e}")
            return {"error": f"Could not load model for request_id {request_id}: {str(e)}"}
        except Exception as e:
            print(f"🔍 [Inference Debug] Unexpected error: {e}")
            return {"error": f"An unexpected error occurred during model download for {request_id}: {str(e)}"}
        
        # Load base model and apply LoRA adapters
        try:
            # Check what files are actually available
            print(f"🔍 [Inference Debug] Available files in {MODEL_DIR}:")
            if os.path.exists(MODEL_DIR):
                for item in os.listdir(MODEL_DIR):
                    print(f"🔍 [Inference Debug]   {item}")
            else:
                print(f"🔍 [Inference Debug] MODEL_DIR {MODEL_DIR} does not exist!")
            
            # Create missing optional files BEFORE loading tokenizer
            optional_files = {
                'added_tokens.json': '{}',
                'special_tokens_map.json': '{"bos_token": null, "eos_token": null, "unk_token": null, "pad_token": null}'
            }
            
            for filename, default_content in optional_files.items():
                file_path = os.path.join(MODEL_DIR, filename)
                if not os.path.exists(file_path):
                    print(f"🔍 [Inference Debug] Creating missing optional file: {filename}")
                    with open(file_path, 'w') as f:
                        f.write(default_content)
            
            # Load tokenizer - handle missing optional files gracefully
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            except Exception as tokenizer_error:
                print(f"🔍 [Inference Debug] Tokenizer loading error: {tokenizer_error}")
                # Try loading without optional files
                tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
            
            # Load base model with the same quantization settings as training
            # This matches the LOAD_IN_4BIT = True setting from finetuning
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            base_model_instance = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Apply LoRA adapters
            model = PeftModel.from_pretrained(base_model_instance, MODEL_DIR)
            print(f"Successfully loaded LoRA adapters for request_id {request_id}")
            
        except Exception as e:
            print(f"🔍 [Inference Debug] Error loading model/tokenizer for request_id {request_id}: {str(e)}")
            print(f"🔍 [Inference Debug] Error type: {type(e)}")
            print(f"🔍 [Inference Debug] Error traceback:")
            import traceback
            traceback.print_exc()
            return {"error": f"Error loading model or tokenizer: {str(e)}"}
        
        # Generate response
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            model = model.to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response_text}
        
    except Exception as e:
        return {"error": str(e)}
