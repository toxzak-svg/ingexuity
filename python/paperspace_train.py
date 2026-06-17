#!/usr/bin/env python3
# ============================================================================
# paperspace_train.py — Run Llama 3.2 LoRA training on Paperspace
# ============================================================================
# Usage on Paperspace:
# 1. Create a new notebook with free GPU (M4000 is good, or RTX 4000)
# 2. Upload train_llama.py and your training data
# 3. Set runtime to Python 3.10 with PyTorch
# 4. Run this script
# ============================================================================
# Paperspace Free GPU specs:
# - M4000: 8GB VRAM, 30GB storage
# - RTX 4000: 8GB VRAM, 50GB storage
# ============================================================================

import os
import sys

# Add python directory to path
sys.path.insert(0, '.')

from train_llama import TrainingConfig, load_base_model, setup_lora, load_dataset, train

def main():
    # Paperspace storage path
    DATA_PATH = "./train_data.jsonl"
    OUTPUT_DIR = "/storage/llama3_finetuned"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configuration optimized for Paperspace M4000/RTX4000 (8GB VRAM)
    # Slightly smaller batch size than Kaggle due to 8GB vs 16GB
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=2,  # Smaller batch for 8GB GPU
        gradient_accumulation_steps=16,  # More accumulation to compensate
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
        print(f"[Paperspace] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Paperspace] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and setup LoRA
    print("[Paperspace] Loading model...")
    model, tokenizer = load_base_model(config)
    model = setup_lora(model, config)
    
    # Load dataset
    print(f"[Paperspace] Loading data from: {DATA_PATH}")
    train_texts = load_dataset(DATA_PATH, tokenizer, config)
    
    # Train
    print("[Paperspace] Starting training...")
    final_path = train(model, tokenizer, train_texts, config)
    
    print(f"[Paperspace] Training complete! Model saved to: {final_path}")
    
    return final_path

if __name__ == "__main__":
    main()
