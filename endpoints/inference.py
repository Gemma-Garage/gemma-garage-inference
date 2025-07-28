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

@router.post("/inference")
async def hf_inference(request: InferenceRequest):
    print(f"🔍 [Inference Debug] Received request: {request}")
    
    model_name = request.model_name
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens
    
    print(f"🔍 [Inference Debug] Model name: {model_name}")
    print(f"🔍 [Inference Debug] Prompt: {prompt}")
    print(f"🔍 [Inference Debug] Max new tokens: {max_new_tokens}")
    
    if not model_name or not prompt:
        return {"error": "model_name and prompt are required"}
    
    try:
        # Load model and tokenizer directly from Hugging Face Hub
        print(f"🔍 [Inference Debug] Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"🔍 [Inference Debug] Loading model from {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        print(f"🔍 [Inference Debug] Model and tokenizer loaded successfully")
        
        # Generate response
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            model = model.to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔍 [Inference Debug] Generated response: {response_text}")
        return {"response": response_text}
        
    except Exception as e:
        print(f"🔍 [Inference Debug] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error during inference: {str(e)}"}
