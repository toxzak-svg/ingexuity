#!/usr/bin/env python3
"""
drumroll.py — Run inference with the trained TinyLlama LoRA model
Usage: python drumroll.py "Your prompt here"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys
import os

# Model paths
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-chat-v1.0"
LORA_ADAPTER_PATH = "models/trained_model/notebooks/my_weights"

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    try:
        # Try loading as PEFT adapter
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        print("LoRA adapter loaded successfully!")
    except Exception as e:
        print(f"Could not load LoRA adapter: {e}")
        print("Using base model instead...")

    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove input prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response.strip()

def main():
    if len(sys.argv) < 2:
        prompt = "Hello, how are you?"
    else:
        prompt = sys.argv[1]

    model, tokenizer = load_model()

    print(f"\nPrompt: {prompt}\n")
    print("Generating...")
    response = generate(model, tokenizer, prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()