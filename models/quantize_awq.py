"""
AWQ Quantization Pipeline for IngExuity Models

This script:
1. Downloads the base Llama 3.2 model
2. Applies your LoRA adapter
3. Runs AWQ calibration and quantization
4. Saves an ultra-compact quantized model

Requirements:
    pip install llm-awq transformers peft accelerate

Usage:
    python quantize_awq.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from awq import AutoAWQForCausalLM
from awq.quantize import quantize
from awq.calib import get_calib_dataloader
import os
import shutil

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./trained_model/notebooks/my_weights"
OUTPUT_PATH = "./llama3_awq_quantized"
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 3,  # 3-bit quantization - very aggressive
    "version": "GEMM",
}
CALIB_DATA = "./calibration_data"
CALIB_SAMPLES = 128
SEQ_LEN = 512


def setup_directories():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CALIB_DATA, exist_ok=True)


def load_base_model_and_apply_adapter():
    print("Loading base model and applying LoRA adapter...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    
    print(f"Model loaded and adapter merged. Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


def prepare_calibration_data(model, tokenizer):
    print("Preparing calibration dataset...")
    
    from datasets import load_dataset
    
    # Use a small subset of a public dataset for calibration
    # You can replace this with your own fine-tuning data
    try:
        dataset = load_dataset("teknium/openhermes", split="train[:512]")
    except:
        print("Using wikitext for calibration (fallback)...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:512]")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.save_to_disk(CALIB_DATA)
    print(f"Calibration data saved to {CALIB_DATA}")
    return dataset


def run_awq_quantization(model, tokenizer):
    print("Running AWQ quantization...")
    print(f"Config: {QUANT_CONFIG}")
    
    # Quantize
    quant_model = quantize(model, tokenizer, quant_config=QUANT_CONFIG)
    
    # Save quantized model
    quant_model.save_pretrained(OUTPUT_PATH)
    print(f"AWQ model saved to {OUTPUT_PATH}")


def estimate_size():
    from awq.utils import get_model_size
    size_gb = get_model_size(OUTPUT_PATH) / (1024 ** 3)
    print(f"Quantized model size: {size_gb:.2f} GB")
    return size_gb


def main():
    setup_directories()
    
    model, tokenizer = load_base_model_and_apply_adapter()
    prepare_calibration_data(model, tokenizer)
    run_awq_quantization(model, tokenizer)
    size = estimate_size()
    
    print("\n" + "="*50)
    print("DONE! Quantization complete.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {size:.2f} GB")
    print("="*50)


if __name__ == "__main__":
    main()
