#!/usr/bin/env python3
# ============================================================================
# kaggle_train.py — Run Llama 3.2 LoRA training on Kaggle
# ============================================================================
# Usage on Kaggle:
# 1. Upload train_llama.py and requirements.txt to your Kaggle dataset
# 2. Create a new Notebook with GPU accelerator (P100)
# 3. Upload your training data as a Kaggle dataset
# 4. Run this script or import train_llama directly
# ============================================================================

import os
import sys

# Add python directory to path
sys.path.insert(0, '/path/to/your/dataset')

# Import and run training
from train_llama import TrainingConfig, load_base_model, setup_lora, load_dataset, train

def main():
    # Kaggle dataset path (adjust to your dataset name)
    DATA_PATH = "/kaggle/input/your-training-data/train.jsonl"
    OUTPUT_DIR = "/kaggle/working/llama3_finetuned"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configuration optimized for Kaggle P100 (16GB VRAM)
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=512,
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_flash_attention=True,
        use_gradient_checkpointing=True,
        mixed_precision=True
    )
    
    # Print GPU info
    import torch
    if torch.cuda.is_available():
        print(f"[Kaggle] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Kaggle] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and setup LoRA
    print("[Kaggle] Loading model...")
    model, tokenizer = load_base_model(config)
    model = setup_lora(model, config)
    
    # Load dataset
    print(f"[Kaggle] Loading data from: {DATA_PATH}")
    train_texts = load_dataset(DATA_PATH, tokenizer, config)
    
    # Train
    print("[Kaggle] Starting training...")
    final_path = train(model, tokenizer, train_texts, config)
    
    print(f"[Kaggle] Training complete! Model saved to: {final_path}")
    
    # Download link for fine-tuned LoRA weights
    print("[Kaggle] Your fine-tuned LoRA weights are in:")
    print(f"[Kaggle]   {OUTPUT_DIR}/final/")
    
    return final_path

if __name__ == "__main__":
    main()
