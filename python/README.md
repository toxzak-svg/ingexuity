# Python Scripts for IngExuity

This directory contains Python scripts for model training and inference.

## Scripts

| Script | Purpose |
|--------|---------|
| [`gemma_e2b_service.py`](gemma_e2b_service.py) | HTTP service for Gemma 4 E2B with function calling |
| [`train_tinyllama.py`](train_tinyllama.py) | LoRA fine-tuning for TinyLlama 1.1B |
| [`kaggle_train.py`](kaggle_train.py) | Kaggle-specific training runner (P100 GPU) |
| [`paperspace_train.py`](paperspace_train.py) | Paperspace-specific training runner (M4000/RTX4000) |
| [`lora_config.yaml`](lora_config.yaml) | LoRA hyperparameters configuration |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Fine-tune TinyLlama on Kaggle

1. Create a Kaggle account and enable GPU (P100) in Notebook settings
2. Upload your training data as a Kaggle dataset (JSONL format)
3. Upload `train_tinyllama.py` and `requirements.txt`
4. Run:
   ```python
   from kaggle_train import main
   main()
   ```

### 2. Fine-tune TinyLlama on Paperspace

1. Create a Paperspace account with free GPU tier
2. Upload training data and scripts to your workspace
3. Run:
   ```bash
   python paperspace_train.py
   ```

### 3. Use Fine-tuned Model in Julia

```julia
using IngExuity
using LlamaInference

# Load the base model
load_llama_model()

# Or load your custom fine-tuned model
load_finetuned_model("/path/to/your/finetuned/model.gguf")

# Generate text
result = chat_llama("Hello, how are you?")
println(result["text"])
```

## Training Data Format

Training data should be in JSONL format:

```jsonl
{"text": "Your training example here"}
{"text": "Another example"}
```

Or JSON format:

```json
{
  "data": [
    "First example",
    "Second example"
  ]
}
```

## LoRA Configuration

Default settings optimized for 16GB GPU (Kaggle P100):
- LoRA rank: 32
- Batch size: 4
- Learning rate: 2e-4
- Epochs: 3
- Max sequence length: 512

For 8GB GPU (Paperspace M4000), reduce batch size to 2 and increase gradient accumulation to 16.

## Output

Fine-tuned models are saved to:
- `./tinyllama_finetuned/final/` (local)
- `/kaggle/working/tinyllama_finetuned/final/` (Kaggle)
- `/storage/tinyllama_finetuned/final/` (Paperspace)

## Notes

- TinyLlama is ~1.1B parameters, quantized Q4_K_M is ~634MB
- LoRA training only updates ~0.5-2% of parameters
- Training with QLoRA (4-bit) requires ~6GB VRAM
- Full fine-tuning is possible with ~16GB VRAM
