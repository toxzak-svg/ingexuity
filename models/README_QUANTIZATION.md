# IngExuity Quantization Methods

This directory contains tools for quantizing models to ultra-small sizes for mobile/offline deployment.

## Files

| File | Purpose |
|------|---------|
| `quantize_awq.py` | Full AWQ quantization pipeline |
| `requantize_gguf.py` | Re-quantize existing GGUF to smaller format |
| `requirements_quant.txt` | Python dependencies |
| `awq_setup.py` | Documentation and setup guide |

## Quick Start

### Option 1: AWQ Quantization (Recommended)

AWQ (Activation-aware Weight Quantization) produces better quality than standard quantization at the same size.

```bash
# Install dependencies
pip install -r requirements_quant.txt

# Run quantization
python quantize_awq.py
```

### Option 2: Re-quantize Existing GGUF

If you have llama.cpp installed, you can re-quantize your current model to smaller formats.

```bash
# Build llama.cpp (for quantize tool)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. && cmake --build . --config Release

# Re-quantize
python requantize_gguf.py --all
```

## Size Comparison (Llama 3.2 1B)

| Format | Size | Quality |
|--------|------|---------|
| FP16 | 2.2 GB | Baseline |
| Q4_K_M | 700 MB | Good |
| Q3_K | 450 MB | Acceptable |
| Q2_K | 350 MB | Lower |
| AWQ 3-bit | 500 MB | ~Q4 quality but smaller |

## Memory Targets

IngExuity target: 30-60 MB for mobile deployment.

**Problem:** Even Q2_K (~350MB) is ~5x larger than target.

**Solutions for hitting 30-60MB:**

1. **Use a smaller base model**
   - Llama 3.2 1B is already one of the smallest capable models
   - Consider: custom distilled models or quantization to 2-bit

2. **Heavy pruning + quantization**
   - Prune 70-80% of weights, then quantize what remains
   - This is experimental but can yield very small models

3. **Knowledge distillation**
   - Distill knowledge from Llama 3.2 into a much smaller model
   - Train a 50-100M param model from scratch using Llama 3.2 as teacher

4. **Embedding compression**
   - Embeddings are a significant portion of model size
   - Reduce vocabulary or compress embeddings with SVD

## Recommended Approach

For best quality at smallest size:

1. Start with Llama 3.2 1B
2. Apply your LoRA adapter
3. Use AWQ 3-bit quantization
4. Prune non-critical weights
5. Compress vocabulary if possible

This should yield a ~300-400MB model with quality comparable to Q4_K_M at 700MB.

## Next Steps

See `awq_setup.py` for detailed documentation on:
- Setting up AWQ calibration
- Loading pre-quantized models from HuggingFace
- Training custom AWQ models
