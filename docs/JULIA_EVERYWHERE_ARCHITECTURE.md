# IngExuity Julia-Everywhere Architecture

## Purpose

This document defines the architecture of the Julia implementation line. It separates the stable runtime contract from experimental cognition and model research.

The Rust line on `main` is a separate implementation. Shared concepts should be exchanged through fixtures, schemas, and behavioral tests rather than by treating one codebase as generated output from the other.

## Architectural layers

### 1. Observations

Observations are immutable records of what entered the system or what happened after an intervention.

Examples:

- user message;
- explicit preference correction;
- session lifecycle event;
- tool result;
- response delivery result;
- later user reaction.

An observation must record its source and time. Inferred emotional or user-model state is not an observation.

### 2. Belief state

Belief state contains uncertain interpretations of observations:

- user model;
- self model;
- emotional state;
- active topics;
- memory claims;
- prediction ledger;
- capability state.

Belief updates must preserve provenance and uncertainty. A new inference must not silently overwrite a contradictory prior claim.

### 3. Prediction

Prediction is a time-bounded, falsifiable statement issued using a declared evidence set.

Prediction creation and outcome resolution are separate operations. Candidate examples include:

- likely next-turn intent;
- whether acknowledgment is preferred before advice;
- whether the user will continue the current topic;
- whether a proposed response will cause clarification or disengagement.

Predictions must support abstention.

### 4. Intervention policy

The intervention policy generates and ranks candidate actions:

- answer directly;
- acknowledge and stay present;
- ask a clarifying question;
- retrieve memory;
- conduct research;
- abstain;
- take an authorized tool action.

SANDBOX SIM belongs here. It must evaluate candidate interventions with independent or held-out evidence where possible.

### 5. Inference backend

Inference backends provide text or structured inference but do not own identity state.

Target multiple-dispatch contract:

```julia
abstract type AbstractInferenceBackend end

backend_metadata(::AbstractInferenceBackend)::NamedTuple
generate(::AbstractInferenceBackend, request)::GenerationResult
health(::AbstractInferenceBackend)::BackendHealth
```

Planned backend types:

- `LlamaCppBackend` for local GGUF controls;
- `JuliaTransformerBackend` for the conventional Julia-native baseline;
- `LinguistLSABackend` for novel operator experiments;
- `DeterministicBackend` for tests and replay.

No backend becomes default until it passes the same conformance suite.

### 6. Durable store

The durable store owns:

- identity and session records;
- observations;
- events;
- snapshots;
- memory claims and provenance;
- predictions and outcomes;
- model/backend metadata associated with an intervention.

SQLite is the initial implementation target. Storage operations should be transactional and version checked.

### 7. Transport

HTTP, local IPC, and any future UI are transport adapters. They must not own cognitive state or prediction scoring.

## Julia-specific research boundaries

### Multiple dispatch

Multiple dispatch should express real semantic variation, such as:

- backend implementation;
- numeric representation;
- policy evaluator;
- storage adapter;
- device-specific kernel.

It should not be used to hide global mutable state or create ambiguous fallback behavior. Generic fallback methods must be tested explicitly.

### Numeric genericity

Model operators should be written against declared numeric contracts:

```julia
attention(::AbstractMatrix{T}, params::AttentionParams{T}) where {T<:AbstractFloat}
quantized_attention(::AbstractMatrix{Ti}, params::QuantizedParams{Ti,Ts}) where {Ti<:Integer,Ts<:AbstractFloat}
```

Integer inference requires explicit scale, zero point, accumulator width, rounding, and saturation policy. “Julia supports integers” is not enough evidence.

### Compilation

Julia JIT, sysimages, native application compilation, and device targets are separate concerns. A successful desktop package load does not demonstrate mobile or WASM feasibility.

## Model program

### Conventional baseline

The first trained Julia-native model must be conventional enough to validate the stack:

- byte/BPE tokenizer with reference fixtures;
- causal transformer;
- stable masking and normalization;
- reproducible initialization;
- training and held-out evaluation;
- checkpoint manifest and checksum.

### Linguist-LSA

Linguist-LSA is a research program, not the baseline. Its components are evaluated independently:

- selective state-space memory;
- linear-merge attention;
- low-rank SwiGLU;
- quantized/integer kernels.

Each component needs a reference implementation, benchmark, quality comparison, and ablation.

## State transition contract

The target turn operation is transactional:

```text
load versioned session snapshot
  -> validate input
  -> append immutable observation
  -> derive staged belief state
  -> issue predictions
  -> generate intervention candidates
  -> evaluate candidates
  -> invoke selected backend/tool
  -> append output event
  -> atomically commit events + next snapshot
```

If inference, evaluation, or persistence fails, the committed snapshot remains unchanged.

## Prediction and evaluation contract

A valid prediction includes:

- identifier;
- target;
- issue time;
- horizon;
- probability/confidence;
- evidence cutoff;
- resolution rule;
- status;
- eventual outcome and score.

A valid experiment reports:

- dataset or fixture version;
- baseline;
- metric;
- confidence interval where appropriate;
- failure cases;
- hardware/software environment for performance claims.

## Privacy and control

The Julia line remains local-first by design, but local storage still requires controls:

- inspect learned claims;
- correct or invalidate claims;
- delete identity state;
- export/import through a versioned format;
- configure retention;
- disable emotional adaptation;
- disable memory classes or personalization entirely.

## Current known gaps

The pre-Phase-0 prototype has the following known gaps:

- one global conversation state;
- one process-local memory store;
- no durable identity schema;
- direct LLM responses can self-confirm predictions;
- retry logic can increase confidence without new evidence;
- SANDBOX SIM uses overlapping proposal and evaluation signals;
- handwritten transformer defaults to random weights;
- tokenizer behavior is not reference-conformant;
- model and mobile performance claims are not supported by checked-in benchmarks.

These gaps are migration inputs, not reasons to discard the Julia architecture.

## Component maturity labels

Every major component must use one of four labels:

- **active** — used by the default runtime and covered by tests;
- **experimental** — executable and tested, but not default or validated for product claims;
- **specified** — design exists without sufficient implementation;
- **unsupported** — retained for provenance but not maintained.

## Near-term target layout

```text
src/
  IngExuity.jl
  runtime/
    sessions.jl
    transitions.jl
    backends.jl
  cognition/
    observations.jl
    beliefs.jl
    predictions.jl
    policy.jl
  models/
    baseline_transformer/
    linguist_lsa/
    quantization/
  persistence/
    sqlite.jl
    migrations/
  transport/
    http.jl
  legacy/
    modules/
```

The current module tree will be migrated incrementally behind tests rather than rearranged all at once.
