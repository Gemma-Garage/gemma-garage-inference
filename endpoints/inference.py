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
    base_model: str = "unsloth/gemma-3-1b-it"  # Base model for LoRA adapters

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
    
    if not model_name or not prompt:
        return {"error": "model_name and prompt are required"}
    
    try:
        # Check if the model has adapter_config.json (indicating LoRA adapters)
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files(model_name)
            has_adapter_config = "adapter_config.json" in files
            print(f"🔍 [Inference Debug] Model files: {files}")
            print(f"🔍 [Inference Debug] Has adapter_config.json: {has_adapter_config}")
            
            if has_adapter_config:
                # This is a LoRA adapter model
                print(f"🔍 [Inference Debug] Loading as LoRA adapters...")
                from peft import PeftModel
                
                # Load base model
                print(f"🔍 [Inference Debug] Loading base model: {base_model}")
                base_model_instance = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                
                # Apply LoRA adapters
                print(f"🔍 [Inference Debug] Applying LoRA adapters from: {model_name}")
                model = PeftModel.from_pretrained(base_model_instance, model_name)
                print(f"🔍 [Inference Debug] Loaded as LoRA adapters successfully")
            else:
                # This is a regular model
                print(f"🔍 [Inference Debug] Loading as regular model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print(f"🔍 [Inference Debug] Loaded as regular model successfully")
                
        except Exception as e:
            print(f"🔍 [Inference Debug] Error checking model structure: {e}")
            # Fallback: try regular model first, then LoRA
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print(f"🔍 [Inference Debug] Loaded as regular model successfully")
            except Exception as regular_error:
                print(f"🔍 [Inference Debug] Failed to load as regular model: {regular_error}")
                print(f"🔍 [Inference Debug] Attempting to load as LoRA adapters...")
                
                from peft import PeftModel
                
                # Load base model
                print(f"🔍 [Inference Debug] Loading base model: {base_model}")
                base_model_instance = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                
                # Apply LoRA adapters
                print(f"🔍 [Inference Debug] Applying LoRA adapters from: {model_name}")
                model = PeftModel.from_pretrained(base_model_instance, model_name)
                print(f"🔍 [Inference Debug] Loaded as LoRA adapters successfully")
        
        # Load tokenizer from Hugging Face Hub
        print(f"🔍 [Inference Debug] Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"🔍 [Inference Debug] Model and tokenizer loaded successfully")
        
        # Generate response - force everything to CUDA
        device = "cuda"
        print(f"🔍 [Inference Debug] Using device: {device}")
        
        # Ensure model is on CUDA
        model = model.to(device)
        print(f"🔍 [Inference Debug] Model moved to {device}")
        
        # Tokenize and move inputs to CUDA
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"🔍 [Inference Debug] Inputs moved to {device}")
        
        # Generate with proper device handling
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔍 [Inference Debug] Generated response: {response_text}")
        return {"response": response_text}
        
    except Exception as e:
        print(f"🔍 [Inference Debug] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error during inference: {str(e)}"}
        