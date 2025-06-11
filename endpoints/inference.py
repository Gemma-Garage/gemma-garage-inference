import os
from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from google.cloud import storage
from peft import PeftModel  # Added import

router = APIRouter()

MODEL_DIR = "/app/model"  # Local directory to download adapter files to
GCS_BUCKET = "llm-garage-models"
# GCS_PREFIX should point to the parent folder of specific request_id folders
GCS_PREFIX = "gemma-peft-vertex-output/model"

class PromptRequest(BaseModel):
    prompt: str
    request_id: str
    base_model:str 


def download_model_from_gcs(bucket_name, gcs_prefix_for_model, local_dir):  # Renamed prefix to gcs_prefix_for_model for clarity
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure local_dir is clean before downloading new model files
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
        # Ensure blob.name is not ending with '/' (which indicates a directory itself)
        if blob.name.endswith("/"):
            continue

        # Construct file_path correctly
        # blob.name might be "gemma-peft-vertex-output/model/request_id/final_model/file.json"
        # gcs_prefix_for_model is "gemma-peft-vertex-output/model/request_id/final_model/"
        # We want the part after gcs_prefix_for_model, e.g., "file.json"
        relative_blob_path = blob.name[len(gcs_prefix_for_model):]
        if relative_blob_path.startswith("/"):  # Remove leading slash if any
            relative_blob_path = relative_blob_path[1:]

        file_path = os.path.join(local_dir, relative_blob_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
    print(f"Downloaded model files from gs://{bucket_name}/{gcs_prefix_for_model} to {local_dir}")


# Download model at startup
@router.post("/predict")
def predict(request: PromptRequest):
    request_id = request.request_id
    base_model = request.base_model
    # This is the GCS path for the specific adapter model files
    adapter_gcs_path = f"{GCS_PREFIX}/{request_id}/final_model/"

    try:
        download_model_from_gcs(GCS_BUCKET, adapter_gcs_path, MODEL_DIR)
    except FileNotFoundError as e:
        return {"error": f"Could not load model for request_id {request_id}: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during model download for {request_id}: {str(e)}"}

    try:
        # Load the tokenizer that was saved with the adapter
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("Tokenizer loaded successfully.")
        # Load the base model
        # Using bfloat16 for Gemma models is common for performance and memory
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,  # Or torch.float16 if bfloat16 is not supported
            device_map="auto"  # Automatically distribute model across available devices
        )
        print("Base model loaded successfully.")
        # Apply PEFT adapter to the base model
        # MODEL_DIR now contains the adapter files (adapter_config.json, adapter_model.bin)
        model = PeftModel.from_pretrained(base_model_instance, MODEL_DIR)
        print("Adapter model loaded successfully.")
        # Merge the adapter layers into the base model.
        # This makes inference faster after the one-time cost of merging.
        model = model.merge_and_unload()
        print("Adapter model merged successfully.")
        # Ensure the final model is on the correct device, though device_map="auto" should handle it.
        # Explicitly moving can be a safeguard.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    except Exception as e:
        # Log the full error for debugging
        print(f"Error loading model/tokenizer for request_id {request_id}: {str(e)}")
        return {"error": f"Error loading model or tokenizer: {str(e)}"}

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)  # Ensure inputs are on the same device as model
    outputs = model.generate(**inputs, max_new_tokens=100)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response_text}

# Ensure to remove or comment out any old model loading logic if present,
# especially any global model loading at startup if each request loads a specific model.
