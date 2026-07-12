# IngExuity

IngExuity is a local-first companion research project built around explicit user modeling, durable memory, and falsifiable prediction.

The production substrate is **Rust**. The existing Julia implementation remains in the repository temporarily as legacy behavioral reference while useful concepts and tests are migrated.

> Current status: Phase 2 durable runtime. Session creation, chat, reset, deletion, expiration, and restart recovery are backed by a versioned SQLite event store. Generation remains model-free and deterministic until the inference boundary is ready.

## What is active

| Component | Status | Notes |
|---|---|---|
| Rust workspace | **active** | Root Cargo workspace, locked to Rust 1.97 |
| `ingexuity-core` | **active** | Typed state and transactional turn processing |
| `ingexuity-server` | **active** | Hardened Axum API with durable UUID sessions |
| `ingexuity-store` | **active** | SQLite migrations, snapshots, ordered events, integrity checks |
| Session lifecycle | **active** | Persistence, restart recovery, sliding expiry, reset, deletion |
| Deterministic fallback | **active** | Model-free CI and failure-path backend |
| Rust CI and container | **active** | Strict Rust checks and volume-backed restart smoke test |
| Provenance-aware memory claims | planned | Next Phase 2 slice |
| GGUF/llama.cpp backend | planned | Added after cancellation and conformance contracts |
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
  -> clone committed ConversationState
  -> issue an unresolved prediction
  -> run InferenceBackend
       currently: deterministic model-free fallback
  -> append assistant turn
  -> append typed events + replace SQLite snapshot in one transaction
  -> commit in-memory state only when persistence succeeds
  -> JSON response with x-request-id
```

A failed backend call does not partially mutate session state. A failed SQLite write restores the previous in-memory snapshot. Producing a response does not mark a prediction correct. Optimistic version conflicts return `409 state_conflict` rather than silently overwriting newer state.

## Run locally

Install Rust 1.97, then:

```bash
cargo run --locked -p ingexuity-server
```

The default database is `data/ingexuity.sqlite3`. Override the bind address or database path with:

```bash
INGEXUITY_BIND=127.0.0.1:9000 \
INGEXUITY_DB_PATH=/path/to/identity.sqlite3 \
cargo run --locked -p ingexuity-server
```

Startup creates the database parent directory, applies checked-in migrations, runs SQLite integrity checks, and restores non-expired sessions.

### Liveness and readiness

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

`/health/live` answers whether the Rust process can serve HTTP. `/health/ready` reports inference capacity and SQLite integrity. `/health` remains a liveness alias.

### Create a session

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
```

Sessions use opaque UUIDs and a 30-minute sliding inactivity timeout by default. Creation is acknowledged only after the initial snapshot is durable.

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

Reset clears conversational and personalized state while advancing the durable state version. Delete removes both the snapshot and its event history through a foreign-key cascade.

## API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Backward-compatible liveness alias |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Backend, storage, and capacity readiness |
| `POST` | `/api/v1/sessions` | Create a durable isolated session |
| `GET` | `/api/v1/sessions/{id}` | Read non-sensitive session metadata |
| `POST` | `/api/v1/sessions/{id}/reset` | Durably reset session state |
| `DELETE` | `/api/v1/sessions/{id}` | Delete durable session state and events |
| `POST` | `/api/v1/chat` | Process and durably commit one turn |

Every response carries `x-request-id`. Request tracing records HTTP metadata rather than raw conversation bodies. JSON requests require `application/json`, and the transport rejects oversized bodies before domain processing.

Errors use a stable JSON envelope:

```json
{
  "error": {
    "code": "state_conflict",
    "message": "session state changed concurrently; retry the request"
  }
}
```

Current error codes include `invalid_json`, `unsupported_media_type`, `request_too_large`, `unknown_session`, `server_busy`, `state_conflict`, `storage_unavailable`, `state_version_overflow`, `empty_message`, `message_too_large`, and `inference_unavailable`.

## Test

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo build --locked --workspace --release
```

Tests cover deterministic replay, transactional rollback, session isolation, durable reset/delete/expiry, optimistic conflicts, SQLite migration and integrity, file-backed restart recovery, request validation, capacity bounds, and container restart recovery against a named volume.

## Container

```bash
docker build -t ingexuity .
docker volume create ingexuity-data
docker run --rm -p 8000:8000 \
  -v ingexuity-data:/app/data \
  ingexuity
```

The image runs as a non-root user, stores SQLite data under `/app/data`, and does not download or bundle a language model.

## Current limitations

- Durable sessions currently store the complete private conversation snapshot; provenance-aware memory claims, selective correction, export/import, and retention controls are the next Phase 2 work.
- Session UUIDs isolate state but are not a complete hosted authentication system.
- Concurrency is bounded, but the current synchronous inference trait cannot safely cancel an already-running backend call. Cancellation and timeouts require the planned inference adapter.
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
