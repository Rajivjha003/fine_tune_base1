import unsloth
from unsloth import FastLanguageModel
import sys
import logging
import os

def export_model():
    print("=======================================")
    print("      MerchFine GGUF Export Tool       ")
    print("=======================================")
    
    # 1. Load the model directly from our LoRA output directory
    try:
        print("\n[1/3] Loading base unsloth/gemma-3-4b-it + local LoRA weights...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="outputs/lora_gemma_3_4b", 
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)
        
    print("\n[2/3] Exporting to GGUF Format (q4_k_m quantization)...")
    print("NOTE: This process will take 5-10 minutes and heavily use CPU/RAM.")
    
    output_dir = "outputs/merchfine-gemma-3-4b"
    
    try:
        # Note: Unsloth save_pretrained_gguf will download llama.cpp binary dynamically if it isn't configured
        # It merges the LoRA natively into the base layers, resulting in 1 standalone quantized file.
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")
        
        print("\n[3/3] Export Successful!")
        print(f"GGUF binary generated in: {output_dir}")
    except Exception as e:
        print(f"\nFailed to export GGUF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    export_model()
