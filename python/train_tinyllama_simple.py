#!/usr/bin/env python3
"""
train_tinyllama_simple.py — Simplified TinyLlama LoRA Training
=============================================================
Single script for Kaggle and Paperspace GPU training.

Usage:
    1. Kaggle: Upload this file + requirements.txt + train.jsonl
    2. Run: !python train_tinyllama_simple.py --data train.jsonl

After training, download the output from ./tinyllama_finetuned/final/
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ============================================================================
# Config
# ============================================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
DEFAULT_OUTPUT = "./tinyllama_finetuned"

# ============================================================================
# Device
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def gpu_info():
    if not torch.cuda.is_available():
        return "CPU"
    return f"CUDA:0 ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)"

# ============================================================================
# Model
# ============================================================================

def load_model_tokenizer():
    print(f"[Train] Loading {MODEL_NAME}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if torch.cuda.is_available() else None
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    print(f"[Train] Model loaded on {gpu_info()}")
    return model, tokenizer

# ============================================================================
# LoRA
# ============================================================================

def setup_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Train] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

# ============================================================================
# Dataset
# ============================================================================

class TextDataset:
    def __init__(self, texts, tokenizer, max_len=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

def load_data(path):
    path = Path(path)
    texts = []
    
    if path.suffix == '.jsonl':
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item.get('text', item.get('content', '')))
    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
            texts = data.get('data', data.get('texts', []))
    else:
        raise ValueError(f"Unsupported: {path}")
    
    print(f"[Train] Loaded {len(texts)} examples")
    return texts

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data (JSONL or JSON)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--maxlen", type=int, default=512)
    args = parser.parse_args()
    
    print(f"[Train] Device: {gpu_info()}")
    
    # Load model + LoRA
    model, tokenizer = load_model_tokenizer()
    model = setup_lora(model)
    
    # Load data
    texts = load_data(args.data)
    
    # Dataset
    dataset = TextDataset(texts, tokenizer, args.maxlen)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"[Train] Training for {args.epochs} epochs...")
    trainer.train()
    
    # Save
    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[Train] Done! Saved to: {final_path}")
    
    return final_path

if __name__ == "__main__":
    main()
