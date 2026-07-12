# J1 Transformer Baseline

J1 is the modern Julia-native control architecture for IngExuity model research. It is intentionally separate from `NanoGPT.jl` so the corrected GPT-style implementation remains available as a behavioral and numerical control.

## Default configuration

| Property | Value |
|---|---:|
| Vocabulary | 16,000 |
| Context | 4,096 tokens |
| Width | 512 |
| Layers | 20 |
| Query heads | 8 |
| Key/value heads | 2 |
| Head width | 64 |
| SwiGLU width | 1,365 |
| Stored parameters | 63,252,992 |
| Position representation | RoPE |
| Normalization | RMSNorm |
| Attention | Grouped-query causal attention |
| Biases | Disabled |
| Embedding/output weights | Tied |

At ideal four-bit packing, weights alone would occupy approximately 31.6 MB. That is not a complete runtime-memory estimate: quantization metadata, tokenizer state, activations, the KV cache, and executable/runtime overhead must be measured separately.

## Why this is a control architecture

J1 adopts established small-model components without introducing the proposed Linguist-LSA recurrent mixer. It gives later selective-SSM or hybrid experiments a parameter-counted reference with the same tokenizer, dataset, optimizer, and evaluation budget.

A hybrid model is not considered better unless it beats J1 on predeclared metrics such as validation loss, retrieval, next-state prediction, latency, peak memory, and energy use.

## Incremental decoding contract

`J1Transformer` supports both:

- `forward_all(model, tokens)` for full causal evaluation;
- `decode_step!(model, token, cache)` for one-token incremental decoding.

Tests require the final logits from both paths to agree at every prefix. This protects against RoPE position errors, grouped-head routing mistakes, and stale or malformed key/value state.

The default 4,096-token cache stores two key/value heads per layer rather than eight, reducing cache storage by four relative to ordinary eight-head multi-head attention at the same head width.

## Current status

Implemented and tested:

- RMSNorm;
- rotary position embeddings;
- grouped-query causal attention;
- SwiGLU feed-forward blocks;
- tied token embedding/output projection;
- exact stored-parameter accounting;
- stable cross-entropy evaluation;
- deterministic initialization;
- bounded incremental KV caches;
- full-forward/cache parity tests.

Not yet claimed:

- automatic-differentiation compatibility;
- a complete optimizer or checkpoint format;
- trained language quality;
- integer kernels or quantized inference;
- mobile deployment;
- superiority over NanoGPT or an external baseline.

## Required next experiments

1. Add differentiable, allocation-audited training kernels and verify nonzero finite gradients on every parameter family.
2. Train tiny NanoGPT and J1 models on exactly the same corpus and token budget.
3. Benchmark prefill latency, decode latency, allocations, and cache memory on CPU.
4. Add Int8 and grouped-Int4 linear kernels through Julia multiple dispatch.
5. Implement a parameter-matched recurrent/hybrid candidate only after the J1 measurements are reproducible.
