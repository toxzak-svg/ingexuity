# IngExuity Rust Architecture

**Status:** Active production architecture  
**Decision date:** 2026-07-12  
**Current milestone:** Phase 1 API hardening  
**Legacy reference:** Julia prototype at historical repository paths

## Purpose

IngExuity is being rebuilt as a Rust system whose central claims—personalization, user prediction, memory continuity, and response-policy selection—can be tested independently.

The architecture separates five concerns that the prototype mixed:

1. domain state;
2. session and API boundaries;
3. durable storage;
4. inference;
5. evaluation and outcome resolution.

The active implementation intentionally uses a deterministic model-free backend. This proves state, concurrency, HTTP, and evaluation contracts before a language model can obscure defects.

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

Owns the process and HTTP boundary:

- Axum router and versioned API;
- opaque UUID session creation;
- per-session locking;
- sliding session expiry;
- session metadata, reset, and deletion;
- model-free liveness and backend-aware readiness;
- transport body limits and JSON/content-type enforcement;
- propagated `x-request-id` values;
- private metadata-only HTTP tracing;
- bounded inference admission through a semaphore;
- stable JSON errors;
- API, lifecycle, expiry, concurrency, smoke, and isolation tests.

The server also forbids unsafe code. Future FFI must remain inside a narrow inference adapter crate.

## Active request path

```text
POST /api/v1/chat
  -> assign or propagate x-request-id
  -> trace method/path/status without logging the body
  -> enforce transport body limit
  -> require and deserialize application/json
  -> find opaque session ID
  -> reject and remove expired session
  -> acquire bounded inference permit
  -> acquire only that session's lock
  -> clone ConversationState
  -> append user turn to candidate state
  -> issue unresolved next-turn prediction
  -> call InferenceBackend
  -> append assistant turn
  -> atomically replace committed state
  -> return typed JSON response
```

If inference fails, the cloned candidate is discarded and committed session state remains unchanged. If inference capacity is full, the request receives `429 server_busy` instead of joining an unbounded queue.

## HTTP contract

### Health

- `/health` and `/health/live` are liveness endpoints. They answer whether the Rust process can serve HTTP.
- `/health/ready` reports the actual backend health, active in-memory session count, and available inference permits.
- An unavailable backend makes readiness return `503` while liveness can remain `200`.

### Sessions

- `POST /api/v1/sessions` creates an opaque UUID session.
- `GET /api/v1/sessions/{id}` exposes only lifecycle metadata, not private turns or user-model content.
- `POST /api/v1/sessions/{id}/reset` clears conversational state while preserving the identifier.
- `DELETE /api/v1/sessions/{id}` immediately invalidates the session.
- In-memory sessions use a sliding inactivity timeout. Phase 1 defaults to 30 minutes.

### Requests and errors

- JSON endpoints require `application/json`.
- Bodies are rejected at the transport boundary before domain processing when they exceed the configured limit.
- Every response carries `x-request-id`.
- Errors use a stable envelope with machine-readable codes.
- HTTP tracing records method, URI, latency, status, and request ID; raw conversation bodies are not included by default.

## Core invariants

### Session invariants

- A session ID maps to exactly one mutable state object.
- One session never reads or mutates another session's state.
- A failed request does not increment turn count or state version.
- Session identifiers are opaque and randomly generated.
- Reset produces a fresh `ConversationState` for the same identifier.
- Delete and expiry make future lookup fail.
- Expiry is testable through an injected clock rather than wall-clock sleeps.
- Phase 1 state is ephemeral and must not be described as durable identity.

### Prediction invariants

- Predictions are issued before their target outcome.
- Response generation does not resolve predictions.
- Unresolved predictions remain pending or later expire.
- Confidence cannot increase merely because a response was produced.

### State invariants

- State changes occur through typed functions.
- Backend failure is transactional.
- User text is treated as data, not executable HTML.
- Liveness and readiness have distinct meanings.
- Admission control bounds concurrent backend calls.

## Model-free backend

The deterministic fallback is not intended to simulate intelligence. It exists to:

- keep CI independent of large artifacts;
- provide a truthful failure mode;
- make replay deterministic;
- verify API and state behavior;
- separate runtime correctness from model quality.

Its metadata explicitly identifies it as `deterministic_fallback` and reports no model.

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

Phase 1 bounds how many calls can enter this trait, but it does not claim safe cancellation of an already-running synchronous call. Timeouts that merely abandon a future while native work continues would be misleading and could leak resources.

When cancellation and streaming are implemented, the contract may evolve into an asynchronous or cancellation-aware interface. That change must preserve:

- model-free testing;
- exact metadata;
- bounded resource use;
- explicit timeout/cancellation results;
- no detached runaway inference;
- no hidden state mutation.

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
- bounded generation execution;
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

The intended Phase 2 write model is:

```text
command
  -> validate against current version
  -> produce typed events
  -> append events transactionally
  -> update materialized snapshot
```

This permits replay, audit, migration, and recovery without coupling stored data to compiler layout.

## Security boundary

Phase 1 materially hardens the local API, but it is not yet a public multi-tenant service. Completed controls include:

- transport body limits;
- content-type and JSON validation;
- opaque isolated sessions;
- expiry, reset, and deletion;
- bounded inference admission;
- request IDs;
- metadata-only HTTP tracing;
- stable non-stack-trace errors.

Hosted deployment still requires:

- authentication and authorization;
- rate limiting by authenticated principal or trusted network boundary;
- explicit hosted-mode CORS policy;
- persistence access controls;
- threat modeling for identity import/export;
- fuzzing of request and bundle parsers;
- backend cancellation and timeout guarantees.

Session UUIDs are capability-like references, not a complete identity or authorization system.

## Julia migration rule

The Julia prototype is legacy reference material. A Julia behavior is migrated only when at least one of these is true:

- it encodes a useful domain invariant;
- it supplies a meaningful test case;
- it implements behavior that beats a baseline;
- it documents a product requirement that remains accepted.

Names alone do not justify ports. The Rust design does not recreate twenty modules merely because twenty Julia files exist. No new production-runtime work should be added to the Julia tree.

## Acceptance status

### Phase 0 — complete

- locked Rust formatting, Clippy, tests, and release builds pass in CI;
- the container starts without a model download;
- malformed JSON and unknown sessions produce stable JSON errors;
- concurrent sessions remain isolated;
- the synthetic baseline replays deterministically;
- README and architecture documents label Julia as legacy;
- `Cargo.lock` is committed from a successful clean resolution.

### Phase 1 — current gate

Phase 1 is complete when CI proves:

- liveness and readiness are distinct and truthful;
- every response has a request ID;
- missing media type and oversized bodies are rejected predictably;
- reset, deletion, and sliding expiry have deterministic tests;
- two sessions remain isolated under the hardened router;
- inference admission is bounded and returns `429` at capacity;
- strict Rust and container smoke tests remain green;
- documentation states that state is still ephemeral and cancellation is not yet implemented.
