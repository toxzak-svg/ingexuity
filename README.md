# IngExuity

IngExuity is a local-first companion research project built around explicit user modeling, durable memory, and falsifiable prediction.

The production substrate is **Rust**. The existing Julia implementation remains in the repository temporarily as legacy behavioral reference while useful concepts and tests are migrated.

> Current status: Phase 1 API hardening. The server is model-free and deterministic, with isolated expiring sessions, lifecycle controls, bounded inference concurrency, request IDs, and separate liveness/readiness reporting.

## What is active

| Component | Status | Notes |
|---|---|---|
| Rust workspace | **active** | Root Cargo workspace, locked to Rust 1.97 |
| `ingexuity-core` | **active** | Typed state and transactional turn processing |
| `ingexuity-server` | **active** | Hardened Axum API with isolated UUID sessions |
| Session lifecycle | **active** | Sliding expiry, metadata, reset, and deletion |
| Deterministic fallback | **active** | Model-free CI and failure-path backend |
| Rust CI and container | **active** | Format, Clippy, tests, release build, and live smoke tests |
| GGUF/llama.cpp backend | planned | Added after backend contracts support cancellation and conformance testing |
| SQLite event store | planned | Phase 2; current sessions are memory-only |
| Prediction ledger/replay | planned | Phase 3 |
| Independent SANDBOX SIM | planned | Phase 5 |
| Mobile runtime | research | Requires measured prototypes |
| Julia runtime and models | **legacy** | Reference only; not the production path |

## Why Rust

IngExuity needs explicit ownership of per-user state, safe concurrency, native deployment, durable schemas, deterministic replay, and replaceable local inference backends. Rust provides a strong substrate for those requirements without making the product depend on an unvalidated custom model architecture.

## Current request path

```text
HTTP request
  -> request ID + private metadata-only tracing
  -> transport body limit and JSON/content-type validation
  -> versioned Axum route
  -> opaque session lookup and sliding-expiry check
  -> bounded inference permit
  -> per-session lock
  -> typed ConversationState clone
  -> pending next-turn prediction issued
  -> InferenceBackend
       currently: deterministic model-free fallback
  -> assistant turn appended
  -> state committed atomically
  -> JSON response with x-request-id
```

A failed backend call does not partially mutate session state. Producing a response does not mark a prediction correct. When all inference permits are occupied, the API returns `429 server_busy` rather than accepting unbounded work.

## Run locally

Install Rust 1.97, then:

```bash
cargo run --locked -p ingexuity-server
```

The server binds to `0.0.0.0:8000` by default. Override it with:

```bash
INGEXUITY_BIND=127.0.0.1:9000 cargo run --locked -p ingexuity-server
```

### Liveness and readiness

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

`/health/live` answers whether the Rust process can serve HTTP. `/health/ready` reports the real inference backend state, active in-memory session count, and available inference permits. `/health` remains a liveness alias.

### Create a session

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
```

Sessions use opaque UUIDs and a 30-minute sliding inactivity timeout by default. They are **not durable yet** and disappear when the process restarts.

### Send a message

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"SESSION_UUID","message":"Can you help me test this?"}'
```

The current response comes from the deterministic fallback and says so explicitly. Model-backed generation is not yet advertised as active.

### Inspect, reset, or delete a session

```bash
curl http://127.0.0.1:8000/api/v1/sessions/SESSION_UUID
curl -X POST http://127.0.0.1:8000/api/v1/sessions/SESSION_UUID/reset
curl -X DELETE http://127.0.0.1:8000/api/v1/sessions/SESSION_UUID
```

Reset clears conversational state while preserving the session identifier. Delete immediately invalidates the identifier.

## API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Backward-compatible liveness alias |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Backend and capacity readiness |
| `POST` | `/api/v1/sessions` | Create an isolated expiring session |
| `GET` | `/api/v1/sessions/{id}` | Read non-sensitive session metadata |
| `POST` | `/api/v1/sessions/{id}/reset` | Reset session state |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session state |
| `POST` | `/api/v1/chat` | Process one message inside a session |

Every response carries `x-request-id`. Request tracing records HTTP metadata rather than raw conversation bodies. JSON requests require `application/json`, and the transport rejects bodies above the configured maximum before domain processing.

Errors use a stable JSON envelope:

```json
{
  "error": {
    "code": "unknown_session",
    "message": "session does not exist or has expired"
  }
}
```

Current error codes include `invalid_json`, `unsupported_media_type`, `request_too_large`, `unknown_session`, `server_busy`, `empty_message`, `message_too_large`, and `inference_unavailable`.

## Test

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo build --locked --workspace --release
```

The checked-in synthetic fixture contains no real user data. Tests cover deterministic replay, transactional failure, conflicting sessions, lifecycle reset/delete, deterministic expiry, request IDs, media-type/body limits, readiness, and bounded inference concurrency.

## Container

```bash
docker build -t ingexuity .
docker run --rm -p 8000:8000 ingexuity
```

The image does not download or bundle a language model.

## Current limitations

- Sessions and user models are in memory only; durable identity begins in Phase 2.
- Session UUIDs isolate state but are not a complete hosted authentication system.
- Concurrency is bounded, but the current synchronous inference trait cannot safely cancel an already-running backend call. Cancellation and timeouts require the planned inference-adapter boundary.
- Rate limiting, explicit hosted-mode CORS policy, and authentication remain release blockers for public multi-tenant deployment.
- Prediction quality, SANDBOX SIM value, emotional adaptation, and mobile performance remain unproven research hypotheses.

## Engineering and research plans

- [Rust-native phased build plan](plans/INGEXUITY_PHASED_BUILD_PLAN.md)
- [Expanded research directions](docs/RESEARCH_DIRECTIONS.md)
- [Rust architecture](docs/RUST_ARCHITECTURE.md)

## Legacy Julia prototype

The Julia files currently remain at their historical paths to preserve provenance while migration proceeds. Do not add new product-runtime work to them. A later cleanup will move retained reference material under `legacy/julia/` after Rust parity tests identify what is worth keeping.

## License

MIT
