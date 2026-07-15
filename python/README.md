# Python Scripts for IngExuity

This directory contains Python scripts for model training and inference.

## Scripts

| Script | Purpose |
|--------|---------|
| [`gemma_e2b_service.py`](gemma_e2b_service.py) | HTTP service for Gemma 4 E2B with function calling |
| [`train_llama.py`](train_llama.py) | LoRA fine-tuning for Llama 3.2 1B |
| [`train_llama_simple.py`](train_llama_simple.py) | Simplified LoRA fine-tuning for Llama 3.2 1B |
| [`build_synthetic_dataset.py`](build_synthetic_dataset.py) | Build validated prediction-training smoke data |
| [`train_weights.ipynb`](../models/trained_model/notebooks/train_weights.ipynb) | End-to-end Colab/Kaggle QLoRA notebook |
| [`kaggle_train.py`](kaggle_train.py) | Kaggle-specific training runner (P100 GPU) |
| [`paperspace_train.py`](paperspace_train.py) | Paperspace-specific training runner (M4000/RTX4000) |
| [`lora_config.yaml`](lora_config.yaml) | LoRA hyperparameters configuration |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Recommended: validated synthetic Kaggle smoke

Open [`train_weights.ipynb`](../models/trained_model/notebooks/train_weights.ipynb) in Kaggle, enable an NVIDIA GPU and Internet, add `HF_TOKEN` as a secret, and run all cells. The default synthetic mode builds 100 records and trains for one epoch.

For local dataset validation:

```bash
python -m pytest python/test_ingexuity_data.py python/test_training_notebook.py -v
python python/build_synthetic_dataset.py --output data/synthetic-pilot --count 100 --seed 42
```

See [`docs/training/SYNTHETIC_DATA.md`](../docs/training/SYNTHETIC_DATA.md) for the schema, outputs, and claim boundary. The 100-row run is a pipeline artifact, not the approved 10,000-example pilot and not evidence of improved user prediction.

### 1. Fine-tune Llama 3.2 on Kaggle

1. Create a Kaggle account and enable GPU (P100) in Notebook settings
2. Upload your training data as a Kaggle dataset (JSONL format)
3. Upload `train_llama.py` and `requirements.txt`
4. Run:
   ```python
   from kaggle_train import main
   main()
   ```

### 2. Fine-tune Llama 3.2 on Paperspace

1. Create a Paperspace notebook with free GPU
2. Upload `train_llama.py` and your training data
3. Run:
   ```python
   from paperspace_train import main
   main()
   ```

## After Training

After fine-tuning, you'll get LoRA weights in the output directory (`./llama3_finetuned/final/`).

### Export to GGUF (for Julia integration)

```bash
python export_to_gguf.py
```

This merges the LoRA adapter with the base model and converts to GGUF format for use with `LlamaInference.jl`.

### Test the fine-tuned model

```bash
python drumroll.py "Your prompt here"
```

## Training Output Locations

| Platform | Output Path |
|----------|-------------|
| Local | `./llama3_finetuned/final/` |
| Kaggle | `/kaggle/working/llama3_finetuned/final/` |
| Paperspace | `/storage/llama3_finetuned/final/` |

## Model Info

- Llama 3.2 1B is ~1.0B parameters, quantized Q4_K_M is ~700MB
- 128K context window
- Trained with the same LoRA configuration as the original setup
