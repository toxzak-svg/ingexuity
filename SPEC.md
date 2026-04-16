# IngExuity — Voice Layer Spec

## What It Is

IngExuity is a prediction-first life partner AI built in Julia. Full architecture in `docs/INGEXUITY_ARCHITECTURE.md`.

**This file**: The Mamba/Linguist voice generation layer that replaces the placeholder in `Response.jl`.

## Current Weak Link

`Response.jl` has a toy RNN (VOCAB_SIZE=1000, HIDDEN_SIZE=128) that generates hash-based placeholder responses. This is the bottleneck.

The Mamba fine-tune fixes this.

## Architecture Stack

```
User input → Starfire (AGI: identity + memory + reasoning)
                    ↓
              IngExuity voice (fine-tuned Mamba-370M)
                    ↓
              Response.jl (formulate + adjust_tone)
                    ↓
              Voice.jl (text → speech)
```

## v1: Mamba-370M Fine-Tune

**Goal**: Replace toy RNN in `Response.jl` with a fine-tuned Mamba-370M that speaks in Zach's voice.

**Notebook**: `finetune_mamba_370m.ipynb`

**Pipeline**:
1. Load `state-spaces/mamba-370m` (HuggingFace)
2. QLoRA fine-tune on `ZacharyMaronek/starfire-personal-v1`
3. INT8 quantization (fits Colab T4)
4. Push to `toxzak/ingexuity-370m`
5. Export GGUF for inference

**Julia Integration**:
```julia
# In Response.jl — call the exported model
using IngExuityVoice  # wraps the Mamba pipeline
content = generate_voice(prompt, tone; model=ingexuity_model)
```

**Hosting**: Railway Pro (2GB RAM needed for INT4 Mamba-370M ~185MB)

## v2: Linguist-LSA 500M

Full architecture in `LINGUIST_LSA_500M.md` (to be created).

- 509M params at INT4 = ~260MB
- >30 tok/s on ARM Cortex-A78
- Causal conv (k=3) + selective SSM + GeGLU FFN
- Needs A100 or RTX 5000 to train (100B tokens)

## Data Requirements

- 50M tokens for fine-tune v1 (Colab T4 viable)
- 100B tokens for full LSA v2 (needs GPU cluster)

## Quantization

| Format | Size | Hosting |
|--------|------|---------|
| FP16 | 740MB | Won't fit |
| INT8 | 370MB | Tight |
| INT4 | 185MB | Railway Pro |
