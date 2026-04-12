import unsloth
from unsloth import FastLanguageModel
import sys
import logging
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from threading import Lock

# Suppress warnings
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("MerchFine_Inference")

# Setup FastAPI App
app = FastAPI(title="MerchFine Native Inference API")

# Global State
global_model = None
global_tokenizer = None
model_lock = Lock()

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

@app.on_event("startup")
async def startup_event():
    global global_model, global_tokenizer
    print("=======================================")
    print("    MerchFine Native Inference Backend ")
    print("=======================================")
    print("\nLoading unsloth/gemma-3-4b-it + local LoRA weights into VRAM...")
    try:
        global_model, global_tokenizer = FastLanguageModel.from_pretrained(
            model_name="outputs/lora_gemma_3_4b",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(global_model)
        print("\n✅ System initialized and Ready on RTX 4070!")
        print("Listening for cURL requests on http://127.0.0.1:8000/api/chat")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    global global_model, global_tokenizer
    
    if global_model is None:
        return {"error": "Model is not loaded."}
        
    messages = [
        {"role": "user", "content": [{"type": "text", "text": request.message}]}
    ]
    
    # We must lock the model generation so simultaneous requests don't crash the GPU
    with model_lock:
        inputs = global_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        
        # Note: Unsloth runs extremely fast, but we generate straight to string to return as JSON
        outputs = global_model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=global_tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = global_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    return {
        "message": response_text.strip(),
        "model_used": "merchfine-gemma-3-4b-native"
    }

if __name__ == "__main__":
    if "--api" in sys.argv:
        print("Starting FastAPI Native Backend...")
        uvicorn.run("run_inference:app", host="127.0.0.1", port=8000, log_config=None)
    else:
        # Standard interactive loop
        print("Run 'python run_inference.py --api' to launch the cURL server.")
        import asyncio
        asyncio.run(startup_event())
        
        from transformers import TextStreamer
        text_streamer = TextStreamer(global_tokenizer, skip_prompt=True)
        
        while True:
            try:
                user_input = input("\n[You]> ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                    
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": user_input}]}
                ]
                
                inputs = global_tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
                ).to("cuda")
                
                print("\n[MerchFine]> ", end="")
                _ = global_model.generate(
                    **inputs, streamer=text_streamer, max_new_tokens=512,
                    use_cache=True, temperature=0.3, top_p=0.9, pad_token_id=global_tokenizer.eos_token_id
                )
            except KeyboardInterrupt:
                break
