# IngExuity Rust Architecture

**Status:** Active production architecture  
**Decision date:** 2026-07-12  
**Legacy reference:** Julia prototype at historical repository paths

## Purpose

IngExuity is being rebuilt as a Rust system whose central claims—personalization, user prediction, memory continuity, and response-policy selection—can be tested independently.

The Rust architecture separates five concerns that the prototype currently mixes:

1. domain state;
2. session and API boundaries;
3. durable storage;
4. inference;
5. evaluation and outcome resolution.

The current Phase 0 implementation intentionally uses a deterministic model-free backend. This proves the state and evaluation contracts before a language model can obscure defects.

## Current active crates

### `ingexuity-core`

Owns domain types and deterministic state transitions:

- `ConversationState`;
- `Turn` and `Role`;
- `UserModel`;
- `Prediction` and `PredictionStatus`;
- backend metadata and health;
- `InferenceBackend`;
- transactional `process_turn`;
- deterministic `HeuristicBackend`.

The crate forbids unsafe code. It has no HTTP, database, or llama.cpp dependency.

### `ingexuity-server`

Owns the process and API boundary:

- Axum router;
- opaque UUID session creation;
- per-session locking;
- model-free liveness/readiness information;
- stable JSON errors;
- request validation;
- server smoke and isolation tests.

The server also forbids unsafe code. Future FFI must remain inside a narrow inference adapter crate.

## Active request path

```text
POST /api/v1/chat
  -> deserialize and validate request
  -> find opaque session ID
  -> acquire only that session's lock
  -> clone ConversationState
  -> append user turn to candidate state
  -> issue unresolved next-turn prediction
  -> call InferenceBackend
  -> append assistant turn
  -> atomically replace committed state
  -> return typed JSON response
```

If inference fails, the cloned candidate is discarded and the committed session remains unchanged.

## Core invariants

### Session invariants

- A session ID maps to exactly one mutable state object.
- One session never reads or mutates another session's state.
- A failed request does not increment turn count or state version.
- Session identifiers are opaque and randomly generated.

### Prediction invariants

- Predictions are issued before their target outcome.
- Response generation does not resolve predictions.
- Unresolved predictions remain pending or later expire.
- Confidence cannot increase merely because a response was produced.

### State invariants

- State changes occur through typed functions.
- Backend failure is transactional.
- User text is treated as data, not executable HTML.
- Runtime health reports the actual backend state.

## Model-free backend

The deterministic fallback is not intended to simulate intelligence. It exists to:

- keep CI independent of large artifacts;
- provide a truthful failure mode;
- make replay deterministic;
- verify API and state behavior;
- separate runtime correctness from model quality.

Its metadata explicitly identifies it as `deterministic_fallback` and reports no model.

## Planned crate boundaries

These crates are introduced only when their interfaces are stable enough to justify separation.

### `ingexuity-store`

Responsibilities:

- SQLite connection and migrations;
- event append with optimistic version checks;
- snapshot loading;
- memory claims and provenance;
- deletion and export;
- transaction and corruption handling.

### `ingexuity-inference`

Responsibilities:

- backend registry;
- GGUF/llama.cpp adapter;
- artifact checksum verification;
- cancellation and timeout;
- bounded generation concurrency;
- exact model metadata;
- backend conformance tests.

Unsafe code, if required for FFI, is isolated here and reviewed separately.

### `ingexuity-eval`

Responsibilities:

- deterministic replay;
- prediction resolution;
- baselines;
- Brier/log-loss scoring;
- calibration and coverage;
- ablations;
- versioned evaluation reports.

### `ingexuity-safety`

Responsibilities:

- immutable policy constraints;
- adverse-case fixtures;
- emotional-adaptation controls;
- crisis-policy routing;
- memory-use restrictions for sensitive contexts;
- non-manipulation requirements.

## State evolution

Long-lived identity is not serialized directly from arbitrary Rust structs. Public and durable formats use versioned schemas.

The intended write model is:

```text
command
  -> validate against current version
  -> produce typed events
  -> append events transactionally
  -> update materialized snapshot
```

This permits replay, audit, migration, and recovery without coupling stored data to compiler layout.

## Inference boundary

The current synchronous trait is deliberately narrow:

```rust
pub trait InferenceBackend: Send + Sync {
    fn metadata(&self) -> BackendMetadata;
    fn health(&self) -> BackendHealth;
    fn generate(&self, request: &GenerationRequest)
        -> Result<Generation, InferenceError>;
}
```

When cancellation and streaming are implemented, this contract may evolve into an asynchronous interface. That change must preserve:

- model-free testing;
- exact metadata;
- bounded resource use;
- explicit timeout/cancellation results;
- no hidden state mutation.

## Security boundary

The initial server is not yet a public multi-tenant service. Before hosted deployment it requires:

- authentication/authorization;
- body-size middleware at the transport boundary;
- rate limiting;
- explicit CORS policy;
- request IDs;
- private-by-default logs;
- persistence access controls;
- threat modeling for identity import/export;
- fuzzing of request and bundle parsers.

Session isolation in memory is necessary but not sufficient authentication.

## Julia migration rule

The Julia prototype is legacy reference material. A Julia behavior is migrated only when at least one of these is true:

- it encodes a useful domain invariant;
- it supplies a meaningful test case;
- it implements behavior that beats a baseline;
- it documents a product requirement that remains accepted.

Names alone do not justify ports. The Rust design does not recreate twenty modules merely because twenty Julia files exist.

No new production-runtime work should be added to the Julia tree.

## Near-term acceptance criteria

Phase 0 is complete when:

- Rust formatting, clippy, tests, and release builds pass in CI;
- the container starts without a model download;
- health reports `runtime: rust` and the real backend state;
- malformed JSON and unknown sessions produce stable JSON errors;
- concurrent sessions remain isolated;
- the synthetic baseline replays deterministically;
- README and architecture documents label Julia as legacy;
- `Cargo.lock` is committed from a successful clean resolution.
