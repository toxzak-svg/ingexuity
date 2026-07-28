# Python model tools for IngExuity

This directory contains the supported training, evaluation, inference, and GGUF export utilities for IngExuity.

## Recommended training pipeline

The supported trainer is [`train_llama_simple.py`](train_llama_simple.py). It performs QLoRA supervised fine-tuning with TRL and accepts conversation exports directly.

### Accepted dataset formats

JSONL is recommended, with one object per line.

**Conversation format:**

```json
{"messages":[{"role":"user","content":"How should I approach this?"},{"role":"assistant","content":"Start by separating the decision into..."}]}
```

**Prompt/completion format:**

```json
{"prompt":"Question: ","completion":"Answer"}
```

**Plain text compatibility format:**

```json
{"text":"A complete training example."}
```

For conversations, every assistant turn becomes a separate supervised example. The preceding system, user, tool, and assistant messages become the prompt, while loss is computed only on the new assistant completion. Exact duplicate examples are removed by default.

### Install

Use Python 3.11 or newer on a CUDA training machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r python/requirements.txt
```

Llama 3.2 is gated on Hugging Face. Accept its license and authenticate before training:

```bash
export HF_TOKEN=hf_your_token
```

### Validate the pipeline first

Run a small smoke test before paying for a full GPU run:

```bash
python python/train_llama_simple.py \
  --data /path/to/train.jsonl \
  --output ./runs/ingexuity-smoke \
  --smoke-test 50 \
  --epochs 1 \
  --batch 1 \
  --grad-accum 4
```

### Full 1B QLoRA run

The defaults are intended for a 16 GB or larger NVIDIA GPU:

```bash
python python/train_llama_simple.py \
  --data /path/to/train.jsonl \
  --output ./runs/ingexuity-1b \
  --epochs 3 \
  --batch 4 \
  --grad-accum 8 \
  --max-length 1024
```

Useful switches:

- `--packing` can improve throughput on modern GPUs, but leave it off on older P100-class hardware.
- `--skip-invalid` skips malformed rows instead of failing immediately.
- `--no-4bit` performs non-quantized LoRA loading.
- `--resume-from-checkpoint PATH` resumes an interrupted run.
- `--validation-fraction` and `--test-fraction` default to `0.05` each.

The trainer creates:

```text
runs/ingexuity-1b/
├── final/                     # PEFT LoRA adapter and tokenizer
├── training_manifest.json    # dataset hash, split counts, settings, metrics
├── train_results.json
├── validation_results.json
└── test_results.json
```

## Dataset validation tests

The normalizer has standard-library unit tests:

```bash
PYTHONPATH=python python -m unittest -v python/test_training_data.py
```

## Merge and export to GGUF

`export_to_gguf.py` now performs the actual merge, Hugging Face-to-GGUF conversion, and quantization. It requires a current local `llama.cpp` checkout with `llama-quantize` built.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j

export LLAMA_CPP_DIR="$PWD/llama.cpp"
python python/export_to_gguf.py \
  --model-path ./runs/ingexuity-1b/final \
  --merged-dir ./runs/ingexuity-1b/merged_hf \
  --output ./models/ingexuity-1b.Q4_K_M.gguf \
  --quant-type Q4_K_M
```

The export writes the final GGUF and a neighboring `.gguf.json` provenance file.

## Other scripts

| Script | Purpose |
|---|---|
| [`training_data.py`](training_data.py) | Dataset loading, validation, chat-template rendering, and deduplication |
| [`train_llama_simple.py`](train_llama_simple.py) | Supported conversation-aware QLoRA/SFT trainer |
| [`export_to_gguf.py`](export_to_gguf.py) | Merge the adapter and produce a quantized GGUF |
| [`drumroll.py`](drumroll.py) | Local prompt test utility |
| [`train_llama.py`](train_llama.py) | Older low-level trainer retained for reference |
| [`kaggle_train.py`](kaggle_train.py) | Older Kaggle wrapper; direct CLI training is preferred |
| [`paperspace_train.py`](paperspace_train.py) | Older Paperspace wrapper; direct CLI training is preferred |
