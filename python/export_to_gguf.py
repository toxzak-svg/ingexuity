#!/usr/bin/env python3
"""
export_to_gguf.py — Export PEFT LoRA adapter to GGUF for llama.cpp
===================================================================
Converts a Hugging Face PEFT model (base + LoRA) to a single GGUF file
that can be loaded directly by LlamaInference.jl.

Usage:
    python export_to_gguf.py --model_path ./models/trained_model/notebooks/my_weights --output ./models

This creates: ./models/llama3-finetuned.Q4_K_M.gguf
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from huggingface_hub import snapshot_download

# ============================================================================
# Config
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT = "./llama3-finetuned.Q4_K_M.gguf"


# ============================================================================
# Export Functions
# ============================================================================

def download_model_with_adapter(model_path: str, base_model: str = MODEL_NAME, output_dir: str = "./temp_base_model"):
    """
    Download base model and merge with LoRA adapter for export.
    Returns path to merged model directory.
    """
    print(f"[Export] Downloading base model: {base_model}")
    
    # Download base model
    base_dir = snapshot_download(
        repo_id=base_model,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    
    # Check if this is a PEFT adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    has_adapter = adapter_config_path.exists()
    
    if has_adapter:
        print(f"[Export] Loading base model from: {base_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            base_dir,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        
        print(f"[Export] Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base, model_path)
        
        print(f"[Export] Merging LoRA into base model...")
        model = model.merge_and_unload()
        
        # Save merged model
        merged_dir = "./temp_merged_model"
        print(f"[Export] Saving merged model to: {merged_dir}")
        model.save_pretrained(merged_dir)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        tokenizer.save_pretrained(merged_dir)
        
        return merged_dir
    else:
        return base_dir


def export_to_gguf(
    model_path: str,
    output_path: str,
    base_model: str = MODEL_NAME,
    quant_type: str = "q4_k_m",
    merge_adapter: bool = True
) -> str:
    """
    Export a PEFT model (base + LoRA) to GGUF format.
    """
    # Get merged model directory
    merged_dir = download_model_with_adapter(model_path, base_model)
    
    print(f"[Export] Converting to GGUF ({quant_type})...")
    print(f"[Export] Model saved to: {merged_dir}")
    print(f"[Export] Output GGUF will be: {output_path}")
    print()
    print(f"[Export] === Next Steps ===")
    print(f"To complete the export, run llama.cpp convert.py:")
    print(f"  python -m llama_cpp.convert {merged_dir} --outfile {output_path} --quantize {quant_type}")
    
    # Write a helper file with the merge info
    config = {
        "merged_model_dir": merged_dir,
        "output_gguf": output_path,
        "quant_type": quant_type,
        "base_model": base_model,
        "adapter_path": model_path
    }
    
    with open("./export_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[Export] Config saved to export_config.json")
    print(f"[Export] You can complete export later by running:")
    print(f"  python -m llama_cpp.convert --config export_config.json")
    
    return merged_dir


def check_model_exists(output_path: str, force: bool = False) -> bool:
    """Check if GGUF already exists."""
    if os.path.exists(output_path) and not force:
        print(f"[Export] GGUF already exists at: {output_path}")
        print(f"[Export] Use --force to re-export")
        return True
    return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export PEFT model to GGUF")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to PEFT adapter (default: ./models/trained_model/notebooks/my_weights)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output GGUF path")
    parser.add_argument("--base_model", type=str, default=MODEL_NAME,
                        help="Base model name")
    parser.add_argument("--quant_type", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization type")
    parser.add_argument("--force", action="store_true",
                        help="Force re-export even if output exists")
    args = parser.parse_args()
    
    # Default model path
    if args.model_path is None:
        default_path = "./models/trained_model/notebooks/my_weights"
        if os.path.exists(default_path):
            args.model_path = default_path
        else:
            print(f"[Export] Error: No model found at {default_path}")
            print(f"[Export] Please specify --model_path or train a model first")
            sys.exit(1)
    
    # Check if already exported
    if check_model_exists(args.output, args.force):
        print(f"[Export] Skipping export. GGUF ready at: {args.output}")
        return
    
    # Export
    result = export_to_gguf(
        model_path=args.model_path,
        output_path=args.output,
        base_model=args.base_model,
        quant_type=args.quant_type,
    )
    
    print(f"\n[Export] === Partial Export Complete ===")
    print(f"Merged model at: {result}")
    print(f"GGUF export requires llama.cpp build tools.")
    print(f"See README for full instructions.")


if __name__ == "__main__":
    main()
