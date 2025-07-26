import os
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from google.cloud import storage
from peft import PeftModel
from fastapi import Request

router = APIRouter()

MODEL_DIR = "/app/model"
GCS_BUCKET = "llm-garage-models"
GCS_PREFIX = "gemma-peft-vertex-output/model"

class PromptRequest(BaseModel):
    prompt: str
    request_id: str
    base_model: str

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

@router.post("/predict")
def predict(request: PromptRequest):
    request_id = request.request_id
    base_model = request.base_model
    adapter_gcs_path = f"{GCS_PREFIX}/{request_id}/final_model/"
    try:
        download_model_from_gcs(GCS_BUCKET, adapter_gcs_path, MODEL_DIR)
    except FileNotFoundError as e:
        return {"error": f"Could not load model for request_id {request_id}: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during model download for {request_id}: {str(e)}"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model_instance, MODEL_DIR)
        model = model.merge_and_unload()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        print(f"Error loading model/tokenizer for request_id {request_id}: {str(e)}")
        return {"error": f"Error loading model or tokenizer: {str(e)}"}
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

@router.post("/inference")
async def hf_inference(request: Request):
    body = await request.json()
    model_name = body.get("model_name")
    prompt = body.get("prompt")
    max_new_tokens = body.get("max_new_tokens", 100)
    if not model_name or not prompt:
        return {"error": "model_name and prompt are required"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with the same quantization settings used in training
        # This matches the LOAD_IN_4BIT = True setting from finetuning
        try:
            # First attempt: Load with 4-bit quantization (matching Unsloth training exactly)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Unsloth uses True, not False
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            print(f"Successfully loaded model {model_name} with 4-bit quantization")
            
        except Exception as quant_error:
            print(f"4-bit quantized loading failed for {model_name}: {quant_error}")
            print("Falling back to non-quantized loading...")
            
            # Second attempt: Load without quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_8bit=False,
                load_in_4bit=False,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print(f"Successfully loaded model {model_name} without quantization")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            model = model.to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response_text}
    except Exception as e:
        return {"error": str(e)}
