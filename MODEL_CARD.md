# IngExuity Model Card

## Model Overview

**IngExuity** is a life partner AI that becomes irreplaceable through use, not training. It predicts what users need before they ask, stays present during emotional moments, and learns user patterns over time through accumulated conversation.

**Key principle:** The system doesn't ask "what should I say." It asks "what will the user need in the next 30 seconds?"

---

## Models Used

### 1. Trained LoRA Adapter

**Base model:** meta-llama/Llama-3.2-1B-Instruct (1.0B parameters)

**Fine-tuned adapter location:** `models/trained_model/notebooks/my_weights/`

| Component | Details |
|-----------|---------|
| **Architecture** | LoRA (Low-Rank Adaptation) |
| **Rank (r)** | 32 |
| **Alpha** | 64 |
| **Dropout** | 0.05 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Quantization** | 4-bit (nf4) with double quantization |
| **PEFT version** | 0.6.2 |

**Files:**
- `adapter_model.safetensors` — LoRA weights
- `adapter_config.json` — PEFT configuration
- `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model` — Tokenizer

### 2. Quantized GGUF Models

**Llama 3.2 1B Instruct Q4_K_M:** `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (~700MB)

| Format | Size | Quality |
|--------|------|---------|
| FP16 | 2.2 GB | Baseline |
| Q4_K_M | ~700 MB | Good for general use |
| Q3_K | ~450 MB | Acceptable |
| Q2_K | ~350 MB | Lower quality |

**Target:** 30-60MB for mobile deployment (requires further quantization/pruning)

### 3. Julia-Native Transformer

**NanoGPT.jl** (`src/modules/NanoGPT.jl`) — Pure Julia transformer built with Flux.jl:

- ~50M parameters
- GPT-2 style architecture with pre-norm blocks
- GELU activation
- Autoregressive generation with top-k/top-p sampling
- **No external dependencies** — fully offline

**Linguist-LSA Architecture** (`SPEC.md`):
- Selective SSM (Mamba-style) for long-range memory
- Linear merge attention (no KV cache, constant memory)
- Low-rank SwiGLU FFN (66% parameter reduction)
- Target: INT4 quantized inference (~30-60MB total)

---

## Architecture

### 16 Modules + Memory Layer

```
Input Layer:        HumanInput, ResultsAnalysis
Cognitive:          Comprehension, SelfModel, UserModel, InternalEmotional, Curiosity
Research/Reasoning: Research, CreativeIngenuity, Decision, Precognition
Prediction Engine:  Predictions, SandboxSim (PRIMARY)
Output Layer:       Action, ReactionObservance, Response, Voice, Output, Understanding, Intelligence
Memory Layer:       Memory (validity-window store)
```

### Presencing Check

When stress > 0.6 OR emotional_charge > 0.7 OR valence < -0.3:
- System stays present (acknowledges first, solves after)
- Records stay_present in UserModel
- Turns ends, user continues explaining
- Next input follows normal solve path

### Inference Backends (priority order)

1. **TrainedModel.jl** — LoRA adapter + Llama 3.2 base
2. **IGSDCore.jl** — Tiny INT8 model (32M params) for edge devices
3. **LlamaInference.jl** — GGUF via llama.cpp
4. **NanoGPT.jl** — Pure Julia transformer (no external deps)

---

## Training

### LoRA Fine-tuning

**Script:** `python/train_llama.py`

**Configuration** (`python/lora_config.yaml`):
- LoRA rank: 32
- Batch size: 4
- Learning rate: 2e-4
- Epochs: 3
- Max sequence length: 512
- Optimizer: bitsandbytes 4-bit quantization (nf4)

**Training data format:** JSONL
```jsonl
{"text": "Your training example here"}
{"text": "Another example"}
```

**Runners:**
- `python/kaggle_train.py` — Kaggle P100 GPU
- `python/paperspace_train.py` — Paperspace M4000/RTX4000

### Quantization

**Scripts:**
- `models/quantize_awq.py` — AWQ (Activation-aware Weight Quantization)
- `models/requantize_gguf.py` — Re-quantize existing GGUF to smaller formats
- `models/quantize_gguf_requantize.py` — GGUF requantization pipeline

---

## Hardware Targets

| Device | Expected Performance |
|--------|---------------------|
| MacBook M2 (16GB) | ~25-30 tok/s |
| ARM Mobile (Snapdragon 8 Gen 3) | ~15-20 tok/s |
| CPU laptop (x86, AVX2) | ~8-12 tok/s |

**Memory footprint (INT4):** ~130MB model + ~100MB activations ≈ 230MB total

**Goal:** 30-60MB for mobile via further quantization, pruning, or knowledge distillation.

---

## Deployment

### Railway
- Docker container, one-click deploy
- Auto-detects `Dockerfile`

### Mobile (Phase 4)
- Julia WASM in webview (PWA)
- Offline mode with local micro-model inference
- Same modules as server

---

## API Endpoints

```
POST /api/chat       { "message": "..." } → { "response": "..." }
GET  /api/predict    → { "predictions": [...] }
GET  /api/intelligence → { "accuracy": 0.73, ... }
GET  /api/user_model → { "name": "Human", "topics": [...], ... }
GET  /api/memory     → { "facts_stored": 142, ... }
GET  /health         → "ok"
```

---

## Privacy

- All memory is local. Always.
- Identity state bundle is portable but never leaves the device
- No cloud dependency for core experience
- Multi-instance sync is opt-in and user-controlled

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Julia everywhere |
| Neural Networks | Flux.jl |
| Web Server | Genie.jl |
| LLM Inference | llama.cpp (GGUF), NanoGPT.jl |
| Tokenizer | GPT-2 BPE (ported to Julia) |
| Persistence | SQLite.jl (Phase 2) |
| Mobile | Julia WASM via PackageCompiler |

---

## Project Status

**v1.4** — Julia transformer stack started. Phase 1 in progress.

Full roadmap: `plans/INGEXUITY_PHASED_BUILD_PLAN.md`
Architecture spec: `docs/INGEXUITY_ARCHITECTURE.md`
Transformer spec: `SPEC.md`

---

## Citation

**Project:** IngExuity — A life partner AI that becomes irreplaceable through use, not training.

**Author:** Zach Marone

**License:** MIT
