import os
from fastapi import APIRouter, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

router = APIRouter()

class InferenceRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int = 100
    base_model: str = "google/gemma-2b"  # Base model for LoRA adapters

@router.post("/inference")
async def hf_inference(request: InferenceRequest):
    print(f"🔍 [Inference Debug] Received request: {request}")
    
    model_name = request.model_name
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens
    base_model = request.base_model
    
    print(f"🔍 [Inference Debug] Model name: {model_name}")
    print(f"🔍 [Inference Debug] Prompt: {prompt}")
    print(f"🔍 [Inference Debug] Max new tokens: {max_new_tokens}")
    print(f"🔍 [Inference Debug] Base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Try to load as regular model first
    print(f"🔍 [Inference Debug] Attempting to load model from {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    print(f"🔍 [Inference Debug] Loaded as regular model successfully")
    
    # Generate response
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print(f"🔍 [Inference Debug] No GPU available, using CPU")
        model = model.to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"🔍 [Inference Debug] Generated response: {response_text}")
    return {"response": response_text}
        