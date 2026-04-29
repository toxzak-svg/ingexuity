#!/usr/bin/env python3
# ============================================================================
# train_tinyllama.py — LoRA fine-tuning for TinyLlama 1.1B
# Supports Kaggle GPU (P100) and Paperspace free GPU (M4000)
# ============================================================================

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
    gguf_model_path: Optional[str] = None  # Use GGUF for inference, safetensors for training
    
    # LoRA config
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Optimizer
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Output
    output_dir: str = "./tinyllama_finetuned"
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu, mps
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

# ============================================================================
# Device Detection
# ============================================================================

def get_device() -> torch.device:
    """Detect best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for logging"""
    if not torch.cuda.is_available():
        return {"device": "cpu", "memory_total": 0, "memory_free": 0}
    
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "device": f"cuda:{gpu_id}",
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "free_memory_gb": torch.cuda.memory_reserved(gpu_id) / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}"
    }

# ============================================================================
# Model Loading
# ============================================================================

def load_base_model(config: TrainingConfig):
    """Load TinyLlama base model with LoRA setup"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"[Train] Loading model: {config.model_name}")
    
    # Quantization config for memory efficiency (optional)
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Apply gradient checkpointing to save memory
    if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    print(f"[Train] Model loaded successfully")
    return model, tokenizer

# ============================================================================
# LoRA Setup
# ============================================================================

def setup_lora(model, config: TrainingConfig):
    """Setup LoRA adapters"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kv_cache_setting
    
    # Prepare model for kvcache if needed
    if hasattr(model, 'prepare_model_for_kv_cache_setting'):
        model = prepare_model_for_kv_cache_setting(model)
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(path: str, tokenizer, config: TrainingConfig):
    """Load and format training dataset"""
    from transformers import DataCollatorForLanguageModeling
    
    print(f"[Train] Loading dataset from: {path}")
    
    # Try multiple formats
    data_path = Path(path)
    
    if data_path.suffix == '.jsonl':
        # JSONL format: {"text": "..."} per line
        texts = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                texts.append(item.get('text', item.get('content', '')))
    elif data_path.suffix == '.json':
        # JSON format: {"data": ["text1", "text2", ...]}
        with open(data_path, 'r') as f:
            data = json.load(f)
            texts = data.get('data', data.get('texts', []))
    elif data_path.is_dir():
        # Directory - look for train.* files
        train_files = list(data_path.glob("train.*"))
        if train_files:
            texts = load_dataset(str(train_files[0]), tokenizer, config)
            return texts
        else:
            texts = []
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    
    print(f"[Train] Loaded {len(texts)} examples")
    return texts

def tokenize_function(texts: List[str], tokenizer, config: TrainingConfig):
    """Tokenize texts for training"""
    from transformers import AutoTokenizer
    
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors=None
    )
    
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# ============================================================================
# Training
# ============================================================================

def create_optimizer(model, config: TrainingConfig):
    """Create optimizer with layer-wise learning rate decay"""
    from torch.optim import AdamW
    
    # Layer-wise LR decay
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer

def create_scheduler(optimizer, num_training_steps: int, config: TrainingConfig):
    """Create learning rate scheduler"""
    from transformers import get_cosine_schedule_with_warmup
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

def train(
    model,
    tokenizer,
    train_texts: List[str],
    config: TrainingConfig,
    eval_texts: Optional[List[str]] = None
):
    """Main training loop"""
    from torch.utils.data import Dataset, DataLoader
    from transformers import Trainer, TrainingArguments
    
    # Create dataset
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        def __len__(self):
            return len(self.encodings["input_ids"])
        
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item
    
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.grad_clip,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        fp16=config.mixed_precision and torch.cuda.is_available(),
        dataloader_num_workers=4,
        save_strategy="steps",
        save_total_limit=3,
        report_to=["none"],  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"[Train] Starting training for {config.num_epochs} epochs...")
    print(f"[Train] Total steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)}")
    
    trainer.train()
    
    # Save final model
    final_path = Path(config.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"[Train] Training complete! Model saved to: {final_path}")
    
    return final_path

# ============================================================================
# Export to GGUF (for Julia integration)
# ============================================================================

def export_to_gguf(model_path: Path, output_path: Path):
    """Export fine-tuned model to GGUF format for llama.cpp"""
    print(f"[Export] Converting to GGUF: {model_path} -> {output_path}")
    
    # Note: llama.cpp conversion requires the original model in safetensors format
    # The conversion is done via Python bindings to llama-cpp-python's GGUF export
    
    try:
        from llama_cpp import Llama
        # Create a temporary inference model to trigger conversion
        # This is a placeholder - actual GGUF conversion needs llama.cpp tools
        print("[Export] GGUF export requires llama.cpp tools")
        print("[Export] Run after training: python -m llama_cpp.llama_convert")
    except ImportError:
        print("[Export] llama-cpp-python not installed, skipping GGUF export")
    
    return output_path

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with LoRA")
    
    # Model args
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-chat-v1.0",
                       help="Base model name or path")
    
    # Training args
    parser.add_argument("--data", required=True, help="Training data path (JSONL or JSON)")
    parser.add_argument("--output", default="./tinyllama_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    
    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    
    # Hardware
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 mixed precision")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        mixed_precision=not args.no_fp16
    )
    
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Show GPU info
    gpu_info = get_gpu_info()
    print(f"[Train] Device: {gpu_info}")
    
    # Load model and setup LoRA
    model, tokenizer = load_base_model(config)
    model = setup_lora(model, config)
    
    # Load dataset
    train_texts = load_dataset(args.data, tokenizer, config)
    
    if len(train_texts) == 0:
        print("[Train] Error: No training data loaded")
        sys.exit(1)
    
    # Train
    final_path = train(
        model,
        tokenizer,
        train_texts,
        config
    )
    
    # Export to GGUF
    gguf_path = Path(config.output_dir) / "model.gguf"
    export_to_gguf(final_path, gguf_path)
    
    print(f"[Train] Done! Fine-tuned model at: {final_path}")

if __name__ == "__main__":
    main()
