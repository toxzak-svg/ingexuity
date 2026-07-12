# IngExuity

IngExuity is a local-first companion research project built around explicit user modeling, durable memory, and falsifiable prediction.

The production substrate is **Rust**. The existing Julia implementation remains in the repository temporarily as legacy behavioral reference while useful concepts and tests are migrated.

> Current status: Phase 0 Rust foundation. The default server is model-free, deterministic, and intended to prove the runtime, session, API, and evaluation contracts before local model integration.

## What is active

| Component | Status | Notes |
|---|---|---|
| Rust workspace | **active** | Root Cargo workspace |
| `ingexuity-core` | **active** | Typed state and transactional turn processing |
| `ingexuity-server` | **active** | Axum API with isolated UUID sessions |
| Deterministic fallback | **active** | Model-free CI and failure-path backend |
| Rust CI and container | **active** | Format, clippy, tests, release build, smoke tests |
| GGUF/llama.cpp backend | planned | Added after backend contracts are stable |
| SQLite event store | planned | Phase 2 |
| Prediction ledger/replay | planned | Phase 3 |
| Independent SANDBOX SIM | planned | Phase 5 |
| Mobile runtime | research | Requires measured prototypes |
| Julia runtime and models | **legacy** | Reference only; not the production path |

## Why Rust

IngExuity needs explicit ownership of per-user state, safe concurrency, native deployment, durable schemas, deterministic replay, and replaceable local inference backends. Rust provides a strong substrate for those requirements without making the product depend on an unvalidated custom model architecture.

## Current request path

```text
HTTP request
  -> versioned Axum API
  -> opaque session lookup
  -> per-session lock
  -> typed ConversationState clone
  -> pending next-turn prediction issued
  -> InferenceBackend
       currently: deterministic model-free fallback
  -> assistant turn appended
  -> state committed atomically
  -> JSON response
```

A failed backend call does not partially mutate session state. Producing a response does not mark a prediction correct.

## Run locally

Install a stable Rust toolchain, then:

```bash
cargo run -p ingexuity-server
```

The server binds to `0.0.0.0:8000` by default. Override it with:

```bash
INGEXUITY_BIND=127.0.0.1:9000 cargo run -p ingexuity-server
```

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Create a session

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
```

### Send a message

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"SESSION_UUID","message":"Can you help me test this?"}'
```

The current response comes from the deterministic fallback and says so explicitly. Model-backed generation is not yet advertised as active.

## Test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build --workspace --release
```

The checked-in synthetic fixture contains no real user data and covers ordinary questions, stress-language handling, HTML-as-data, deterministic replay, and conflicting multi-session preferences.

## Container

```bash
docker build -t ingexuity .
docker run --rm -p 8000:8000 ingexuity
```

The image does not download or bundle a language model.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus runtime/backend status |
| `POST` | `/api/v1/sessions` | Create an isolated session |
| `POST` | `/api/v1/chat` | Process one message inside a session |

Errors use a stable JSON envelope:

```json
{
  "error": {
    "code": "unknown_session",
    "message": "session does not exist"
  }
}
```

## Engineering and research plans

- [Rust-native phased build plan](plans/INGEXUITY_PHASED_BUILD_PLAN.md)
- [Expanded research directions](docs/RESEARCH_DIRECTIONS.md)
- [Rust architecture](docs/RUST_ARCHITECTURE.md)

## What is not yet proven

IngExuity does not yet claim that it accurately predicts a user's needs, that SANDBOX SIM improves responses, that emotional adaptation is beneficial, or that a small local model meets mobile quality and latency targets. Those are research hypotheses with explicit baselines, ablations, and acceptance gates.

## Legacy Julia prototype

The Julia files currently remain at their historical paths to preserve provenance while migration proceeds. Do not add new product-runtime work to them. A later cleanup will move the retained reference material under `legacy/julia/` after Rust parity tests identify what is worth keeping.

## License

MIT
