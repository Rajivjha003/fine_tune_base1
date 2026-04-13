"""
GGUF export tool — merges LoRA adapter into base model,
quantizes to GGUF format, and generates an Ollama Modelfile.

Usage:
    python export_gguf.py                           # Default export
    python export_gguf.py --lora-dir outputs/lora_gemma_3_4b
    python export_gguf.py --quant q8_0              # Higher quality quantization
"""

import unsloth
from unsloth import FastLanguageModel
import sys
import logging
import os
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Modelfile template for Ollama ─────────────────────────────────────────

MODELFILE_TEMPLATE = """FROM {gguf_path}

# MerchFine — Retail demand forecasting & inventory planning assistant
TEMPLATE \"\"\"{{{{ if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{ end }}}}{{{{ if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ end }}}}{{{{ .Response }}}}<end_of_turn>\"\"\"

SYSTEM \"\"\"You are MerchFine, an expert AI assistant for retail demand forecasting and inventory planning. You specialize in:
- SKU-level demand forecasting with confidence intervals
- MIO (Months of Inventory Outstanding) calculations and recommendations
- Reorder point and safety stock optimization
- Sell-through analysis and markdown strategies
- Seasonal planning and promotional impact assessment

Always provide specific numbers, calculations, and actionable recommendations.
When data is insufficient, clearly state what additional information is needed.
Never fabricate sales data or inventory figures.\"\"\"

PARAMETER temperature 0.05
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 2048
PARAMETER stop "<end_of_turn>"
"""


def export_model(
    lora_dir: str = "outputs/lora_gemma_3_4b",
    output_dir: str = "outputs/merchfine-gemma-3-4b",
    quantization: str = "q4_k_m",
    max_seq_length: int = 2048,
):
    """
    Export a fine-tuned LoRA model to GGUF format and generate Ollama Modelfile.
    
    Args:
        lora_dir: Path to the LoRA adapter directory.
        output_dir: Where to save the GGUF binary and Modelfile.
        quantization: GGUF quantization method (q4_k_m, q5_k_m, q8_0, f16).
        max_seq_length: Maximum sequence length the model supports.
    """
    print("=======================================")
    print("      MerchFine GGUF Export Tool       ")
    print("=======================================")
    
    # 1. Load the model directly from our LoRA output directory
    try:
        print(f"\n[1/4] Loading base model + LoRA weights from '{lora_dir}'...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_dir, 
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)
        
    print(f"\n[2/4] Exporting to GGUF Format ({quantization} quantization)...")
    print("NOTE: This process will take 5-10 minutes and heavily use CPU/RAM.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Note: Unsloth save_pretrained_gguf will download llama.cpp binary dynamically if it isn't configured
        # It merges the LoRA natively into the base layers, resulting in 1 standalone quantized file.
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
        
        print(f"\n[3/4] GGUF export successful! Binary in: {output_dir}")
    except Exception as e:
        print(f"\nFailed to export GGUF: {e}")
        sys.exit(1)

    # 4. Generate Ollama Modelfile
    print("\n[4/4] Generating Ollama Modelfile...")
    try:
        _generate_modelfile(output_path, quantization)
    except Exception as e:
        print(f"\nWarning: Modelfile generation failed: {e}")
        print("You can manually create a Modelfile later.")


def _generate_modelfile(output_dir: Path, quantization: str) -> None:
    """
    Auto-generate an Ollama Modelfile that points to the exported GGUF.
    
    Searches for the GGUF file in the output directory and creates a 
    Modelfile with the MerchFine system prompt and inference parameters.
    """
    # Find the exported GGUF file
    gguf_files = list(output_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"  Warning: No .gguf file found in {output_dir}")
        # Fall back to expected naming pattern
        gguf_path = str(output_dir / f"unsloth.{quantization.upper()}.gguf")
    else:
        gguf_path = str(gguf_files[0].resolve())
        print(f"  Found GGUF: {gguf_files[0].name}")

    # Write the Modelfile
    modelfile_path = output_dir / "Modelfile"
    modelfile_content = MODELFILE_TEMPLATE.format(gguf_path=gguf_path)
    
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"  Modelfile written to: {modelfile_path}")

    # Print the ollama create command for convenience
    model_tag = f"merchfine:{quantization}"
    print(f"\n{'='*50}")
    print(f"  To register with Ollama, run:")
    print(f"    ollama create {model_tag} -f {modelfile_path.resolve()}")
    print(f"\n  Then test with:")
    print(f"    ollama run {model_tag}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="MerchFine GGUF Export Tool")
    parser.add_argument(
        "--lora-dir", default="outputs/lora_gemma_3_4b",
        help="Path to the LoRA adapter directory (default: outputs/lora_gemma_3_4b)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/merchfine-gemma-3-4b",
        help="Where to save the GGUF binary (default: outputs/merchfine-gemma-3-4b)",
    )
    parser.add_argument(
        "--quant", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Quantization method (default: q4_k_m)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    args = parser.parse_args()
    
    export_model(
        lora_dir=args.lora_dir,
        output_dir=args.output_dir,
        quantization=args.quant,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
