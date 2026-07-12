# IngExuity — Julia Everywhere

IngExuity is a local-first companion-AI research project built around explicit user modeling, durable memory, presencing, and falsifiable prediction.

This branch is the **Julia-native implementation line**. The Rust implementation remains on `main`; Julia development is isolated on `julia-main` and feature branches based on it.

> Current status: executable research prototype. The modular cognitive pipeline and GGUF adapter exist. The Julia-native transformer and tokenizers are experimental and are not yet trained, benchmarked, or safe to advertise as a production language model.

## Why keep a Julia implementation?

The Julia line is not a syntax port of the Rust service. It exists to test ideas that benefit from Julia’s numerical and dispatch model:

- multiple-dispatch policy and backend composition;
- generic floating-point and integer kernels;
- direct experimentation with attention, state-space, and quantized operators;
- one-language model, cognition, evaluation, and application research;
- inspectable model internals without a mandatory Python training runtime.

These are hypotheses to test, not automatic advantages. Every claimed benefit must be demonstrated by tests or benchmarks.

## What is currently in the repository

| Component | Status | Notes |
|---|---|---|
| Cognitive module pipeline | active prototype | Comprehension, user/self models, emotional state, predictions, SANDBOX SIM, response policy |
| Presencing policy | active prototype | Chooses acknowledgment before problem-solving from explicit state thresholds |
| LlamaCpp/GGUF adapter | active but optional | Local external model artifact; not Julia-native model execution |
| `NanoGPT.jl` | experimental | Handwritten Julia transformer; random weights by default |
| `BPETokenizer.jl` | experimental | Pure-Julia tokenizer research; requires conformance work |
| Linguist-LSA | research specification | Selective SSM, linear-merge attention, low-rank FFN concepts |
| Memory | prototype | Process-local validity-window store; not durable yet |
| HTTP service | prototype | Current API is useful for development, not public multi-tenant hosting |
| Mobile/WASM | unproven research | No production mobile artifact exists yet |

## Architectural rule

The defining question is:

> What is the user likely to need next, and what later observation could prove that prediction wrong?

A response is not evidence that a prediction was correct. Prediction issuance, intervention selection, outcome observation, and scoring must remain separate operations.

## Current request path

```text
input
  -> comprehension
  -> user/self/emotional state updates
  -> presencing gate
  -> candidate predictions
  -> SANDBOX SIM policy filtering
  -> action and response policy
  -> reaction observation
  -> delayed outcome evaluation
  -> memory and state update
```

The existing implementation does not yet satisfy every boundary in that diagram. Phase 0 records the gaps and establishes tests before behavior is expanded.

## Run the Julia package

Install Julia 1.12, then:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Start the development server:

```bash
julia --project=. -e 'using IngExuity; IngExuity.start()'
```

The GGUF backend is optional. Tests and the Julia-native model smoke harness must run without downloading a model.

## Julia-native model work

The repository contains two distinct model directions:

1. **NanoGPT baseline** — a small, understandable transformer used to validate tensor layouts, tokenization, deterministic initialization, and generation mechanics.
2. **Linguist-LSA research** — selective state-space memory, linear-merge attention, low-rank feed-forward layers, and integer-oriented inference.

The NanoGPT baseline is not the product architecture. It is the control condition against which new Julia-native operators must be measured.

Minimum evidence required before calling a Julia-native backend usable:

- deterministic forward-pass tests;
- tokenizer round-trip and reference conformance tests;
- trained checkpoint provenance;
- held-out loss and generation evaluation;
- memory and latency measurements on target hardware;
- comparison against a conventional local GGUF baseline;
- failure and numerical-stability tests.

## Branch structure

- `main` — Rust implementation line
- `julia-main` — stable Julia implementation line
- `julia-everywhere/*` — Julia feature and research branches

Changes should not be merged between the two implementation lines merely to claim parity. Shared concepts should be specified through fixtures and behavioral contracts.

## Plans and documentation

- [Julia-everywhere phased build plan](plans/INGEXUITY_PHASED_BUILD_PLAN.md)
- [Julia-everywhere architecture](docs/JULIA_EVERYWHERE_ARCHITECTURE.md)
- [Original cognitive architecture](docs/INGEXUITY_ARCHITECTURE.md)
- [Model research specification](SPEC.md)

## Safety and product boundaries

IngExuity must not optimize for guilt, dependency, exclusivity, or replacing human relationships. The research target is useful longitudinal personalization with user control, correction, deletion, and calibrated uncertainty.

## License

MIT
