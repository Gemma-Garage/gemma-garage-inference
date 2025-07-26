import os
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
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_blob_path = blob.name[len(gcs_prefix_for_model):]
        if relative_blob_path.startswith("/"):
            relative_blob_path = relative_blob_path[1:]
        file_path = os.path.join(local_dir, relative_blob_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
    print(f"Downloaded model files from gs://{bucket_name}/{gcs_prefix_for_model} to {local_dir}")

@router.post("/inference")
async def hf_inference(request: Request):
    body = await request.json()
    request_id = body.get("request_id")  # Get request_id instead of model_name
    prompt = body.get("prompt")
    max_new_tokens = body.get("max_new_tokens", 100)
    base_model = body.get("base_model", "google/gemma-3-1b-pt")  # Default base model
    
    if not request_id or not prompt:
        return {"error": "request_id and prompt are required"}
    
    try:
        # Download LoRA adapters from GCS
        adapter_gcs_path = f"{GCS_PREFIX}/{request_id}/final_model/"
        try:
            download_model_from_gcs(GCS_BUCKET, adapter_gcs_path, MODEL_DIR)
        except FileNotFoundError as e:
            return {"error": f"Could not load model for request_id {request_id}: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred during model download for {request_id}: {str(e)}"}
        
        # Load base model and apply LoRA adapters
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            
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
            print(f"Error loading model/tokenizer for request_id {request_id}: {str(e)}")
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
