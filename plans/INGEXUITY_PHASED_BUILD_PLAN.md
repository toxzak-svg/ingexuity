# IngExuity — Rust-Native Robust Build Plan

**Status:** Accepted engineering direction  
**Updated:** 2026-07-12  
**Production language:** Rust  
**Legacy implementation:** Julia prototype retained temporarily as behavioral reference only  
**Scope:** Build a reliable, measurable, privacy-preserving local companion system whose personalization and prediction claims can be falsified.

---

## 1. Architecture Decision

IngExuity will be implemented as a Rust workspace. The current Julia code is not the production foundation. It remains available only to recover useful behaviors, test cases, and terminology while equivalent Rust components are built and verified.

This decision is based on the project's actual requirements:

- predictable memory and concurrency semantics;
- explicit ownership of per-user state;
- safe local persistence;
- small deployable binaries;
- native desktop and mobile integration;
- mature bindings to llama.cpp and other local runtimes;
- strong type-level interfaces for state transitions;
- deterministic replay and property testing;
- long-lived schema and migration support.

The migration is not a line-for-line port. Heuristic modules are preserved only when they have a clear contract, tests, and measurable value.

### Repository status vocabulary

Every component must be labeled as one of:

- **active** — used by the default Rust runtime and covered by CI;
- **experimental** — runnable but not relied upon by the default product;
- **planned** — specified but not implemented;
- **legacy** — retained only for migration reference;
- **deprecated** — scheduled for removal.

---

## 2. Non-Negotiable Engineering Principles

1. **Rust is the production substrate.** New runtime, state, API, storage, and inference integration work is written in Rust.
2. **Repository truth before product narrative.** README claims must match measured behavior.
3. **No self-confirming metrics.** Producing a response never counts as a correct prediction.
4. **Session isolation by construction.** No process-global mutable user state.
5. **State transitions are explicit.** Core functions receive state and return typed events or updates.
6. **Uncertainty and abstention are valid outcomes.** Confidence cannot rise without evidence.
7. **Every adaptive mechanism needs a baseline and ablation.** Modules survive by measured contribution, not naming.
8. **Local-first still requires security.** Local data needs access control, deletion, migration, integrity checks, and a threat model.
9. **Inference backends are replaceable infrastructure.** IngExuity must not depend on one model or custom architecture.
10. **Safety is a system property.** Emotional adaptation, memory, prediction, and output policy require adverse-case tests.
11. **Identity portability follows stable schemas.** Multi-device work waits for migrations, provenance, and conflict rules.
12. **Claims are earned at acceptance gates.** A phase is complete only when its tests and evidence pass.

---

## 3. Target Rust Workspace

```text
IngExuity/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── ingexuity-core/          # domain types and deterministic state transitions
│   ├── ingexuity-server/        # Axum HTTP API and session boundary
│   ├── ingexuity-store/         # SQLite event/state/memory persistence
│   ├── ingexuity-inference/     # backend traits and llama.cpp adapter
│   ├── ingexuity-eval/          # replay, baselines, scoring, calibration
│   └── ingexuity-safety/        # policy constraints and adverse-case checks
├── fixtures/
│   └── synthetic/               # privacy-safe deterministic conversations
├── migrations/                  # versioned SQLite migrations
├── docs/
│   ├── RUST_ARCHITECTURE.md
│   ├── RESEARCH_DIRECTIONS.md
│   ├── THREAT_MODEL.md
│   └── KNOWN_LIMITATIONS.md
└── legacy/
    └── julia/                    # temporary reference implementation
```

The workspace may begin with only `core` and `server`; crates are separated when stable interfaces exist, not merely to create structure.

### Required interfaces

```rust
pub trait InferenceBackend {
    fn metadata(&self) -> BackendMetadata;
    fn health(&self) -> BackendHealth;
    fn generate(&self, request: GenerationRequest) -> Result<Generation, InferenceError>;
}

pub trait StateStore {
    fn create_session(&self) -> Result<SessionId, StoreError>;
    fn load_session(&self, id: SessionId) -> Result<SessionSnapshot, StoreError>;
    fn append_events(&self, id: SessionId, expected_version: u64, events: &[Event])
        -> Result<u64, StoreError>;
    fn delete_session(&self, id: SessionId) -> Result<(), StoreError>;
}

pub trait PredictionLedger {
    fn issue(&self, prediction: Prediction) -> Result<(), LedgerError>;
    fn resolve(&self, resolution: Resolution) -> Result<(), LedgerError>;
    fn score(&self, query: ScoreQuery) -> Result<ScoreReport, LedgerError>;
}

pub trait PolicyEvaluator {
    fn evaluate(
        &self,
        snapshot: &SessionSnapshot,
        candidates: &[ResponsePolicy],
    ) -> Result<Vec<PolicyScore>, PolicyError>;
}
```

`Clock`, `IdGenerator`, and randomness are injectable so replay tests do not depend on wall-clock time or nondeterministic IDs.

---

# Phase 0 — Rust Foundation and Repository Truth

## Goal

Create a clean Rust workspace that builds, tests, starts, and truthfully describes the migration without loading a model.

## Work

### 0.1 Establish the Rust workspace

- Add a root Cargo workspace.
- Create `ingexuity-core` and `ingexuity-server`.
- Define typed session, turn, user-model, prediction, and backend metadata structures.
- Implement a deterministic model-free backend for CI and failure fallback.
- Keep raw model integration outside `core`.
- Set `#![forbid(unsafe_code)]` in crates that do not require FFI.

### 0.2 Reconcile repository truth

- Rewrite README around the Rust runtime and migration status.
- Mark all Julia source and Julia model paths as legacy.
- Remove claims that Flux, Genie, Julia WASM, or a custom transformer are active production components.
- Document the exact active request path.
- Move unverified mobile, latency, parameter-count, and emotional-effect claims into research hypotheses.

### 0.3 Continuous integration

CI must run:

1. `cargo fmt --check`;
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
3. `cargo test --workspace --all-features`;
4. release build;
5. server health smoke test;
6. malformed and missing-field API tests;
7. model-free fallback test;
8. session-isolation test;
9. Docker build without downloading a model.

### 0.4 Baseline fixture

Check in privacy-safe synthetic conversations covering:

- ordinary information seeking;
- short and ambiguous messages;
- topic changes;
- stress-language false positives;
- emotional disclosure;
- preference correction;
- two users with conflicting preferences;
- unavailable and timed-out inference;
- malicious HTML/script text;
- deletion and reset requests.

### Acceptance gate

- A clean checkout passes Rust CI.
- The server starts without any model artifact.
- Health, session creation, chat, malformed JSON, missing field, and unknown-session behavior are tested.
- Two sessions cannot observe or mutate one another.
- README and architecture documents identify Rust as active and Julia as legacy.
- No build step hides failure.

---

# Phase 1 — Session Isolation and Hardened API

## Goal

Make the server correct under concurrent users, failures, and hostile inputs.

## Work

- Use opaque UUID session identifiers.
- Keep each mutable session behind a per-session lock or actor queue.
- Eliminate hidden process-global user state.
- Version endpoints under `/api/v1/`.
- Define stable request/response schemas and machine-readable error codes.
- Enforce content type and maximum request size.
- Add request IDs and structured logs without raw conversation text by default.
- Bound queues, inference concurrency, output tokens, and request time.
- Return separate liveness and readiness information.
- Render browser output with text nodes, never raw `innerHTML`.
- Add explicit CORS, host-binding, and rate-limit configuration.

### Acceptance gate

- Concurrent session tests show zero state leakage.
- Race/loom tests cover critical state transitions where useful.
- Fuzzed API input cannot panic the server.
- Errors expose neither stack traces nor private state.
- Model failure produces a truthful unavailable or fallback result.

---

# Phase 2 — Event-Sourced Durable Identity

## Goal

Create durable, inspectable, correctable state with provenance.

## Work

Use SQLite initially through a Rust migration system. Store:

- sessions;
- ordered events;
- turns;
- memory claims;
- supporting observations;
- user-model features;
- predictions and outcomes;
- backend metadata;
- consent and retention settings.

A memory claim includes:

- stable ID;
- typed category;
- normalized and display values;
- observation, validity, invalidation, and update timestamps;
- source event IDs;
- confidence and uncertainty reason;
- sensitivity class;
- retention rule;
- contradiction and supersession links.

Users must be able to inspect, correct, invalidate, delete, and export state. Export formats are versioned, checksummed, and independent of Rust's internal struct layout.

### Acceptance gate

- State survives restart without leakage.
- Forward migrations are tested from every released schema.
- Interrupted writes and corruption paths have defined recovery behavior.
- Export -> delete -> import yields a canonically equivalent identity.
- Corrections and deletions affect future retrieval and prediction.
- Every retrieved memory identifies provenance.

---

# Phase 3 — Falsifiable Prediction Ledger

## Goal

Turn “prediction-first” into a measurable subsystem.

Every prediction records:

- prediction and session IDs;
- target variable;
- issue time;
- horizon or deadline;
- finite outcome space or scoring rule;
- probability distribution or confidence;
- evidence available at issue time;
- predictor and policy versions;
- abstention state;
- resolution source;
- outcome and resolution time.

Predictions are stored before response generation and resolved only from later evidence, explicit feedback, or blinded annotation. Unresolved predictions expire; they are never treated as correct.

Baselines include majority class, last-topic persistence, recency-only preference, generic response policy, and the underlying model without personalized state.

Metrics include Brier score/log loss, calibration, coverage, abstention, macro and per-user results, confidence intervals, sample counts, and unresolved rate.

### Acceptance gate

- No score updates without a pre-existing prediction.
- A deterministic replay set compares personalization to named baselines.
- Calibration and coverage accompany accuracy.
- Removing user-specific state is a mandatory ablation.

---

# Phase 4 — Evidence-Weighted User Modeling

## Goal

Learn user-specific patterns without treating each interaction as proof.

Separate:

- explicit user statements;
- observed behavior;
- inferred preferences;
- temporary conversational state;
- long-horizon routines;
- communication-style estimates;
- emotional-context estimates.

Use evidence-weighted online updating, decay, contradiction tracking, correction precedence, and shrinkage for sparse evidence. Confidence does not increase merely because another turn occurred.

### Acceptance gate

- Explicit correction overrides weaker inference.
- Confidence is not monotonic in turn count.
- Personalized policy beats a non-personalized baseline on held-out data.
- False-personalization and stale-memory rates are reported.
- Users can inspect and alter consequential evidence.

---

# Phase 5 — Independent SANDBOX SIM

## Goal

Evaluate response policies independently instead of restating the generator's heuristics.

Evaluate structured choices such as:

- acknowledgment-first vs. solution-first;
- concise vs. detailed;
- ask-one-question vs. answer-now;
- direct vs. hedged;
- retrieve-memory vs. avoid-memory;
- act vs. request confirmation.

The evaluator receives an immutable snapshot and cannot mutate confidence or user state. Candidate generation and evaluation are separate interfaces. Safe opt-in exploration may support contextual-bandit research, but safety constraints dominate learned utility.

### Acceptance gate

- SANDBOX SIM beats no-sandbox and threshold-only baselines on a preregistered metric.
- Evaluator immutability is tested.
- Retry cannot increase confidence without new evidence.
- Negative results and failed policies remain in the evaluation record.

---

# Phase 6 — Emotional Interaction and Safety

## Goal

Make presencing a calibrated, user-controllable policy rather than a keyword-triggered early return.

Treat stress, valence, and emotional charge as uncertain observations. Measure false positives and false negatives. Do not infer stable mental-health conditions from conversational language.

Supported policies include:

- acknowledge and continue solving;
- ask whether the user wants listening or solutions;
- stay with the topic for one turn;
- provide immediate concrete help;
- follow a user-configured default.

Do not optimize guilt, exclusivity, secrecy, dependency, or replacement of human relationships. Retention is evaluated through usefulness, trust, correction success, continuity, and voluntary return.

### Acceptance gate

- Presencing is compared to acknowledgment-plus-help.
- Emotional-intervention false-positive rate is reported.
- Emotional adaptation can be configured or disabled.
- Crisis and adverse-case suites pass deterministic and human-review criteria.

---

# Phase 7 — Local Inference and Edge Deployment

## Goal

Support practical local inference through replaceable Rust backends.

### Reference backend

Use a llama.cpp-compatible GGUF adapter through a maintained Rust binding or a narrow FFI crate. Record:

- model source and immutable revision;
- license;
- SHA-256;
- quantization;
- context size;
- runtime settings;
- hardware and benchmark method.

The backend conformance suite covers load, metadata, generation, cancellation, timeout, unload, corrupt artifact, missing artifact, and bounded concurrency.

### Mobile feasibility

Measure before committing:

- native Rust plus llama.cpp on Android;
- UniFFI/Tauri/NDK integration options;
- local service plus PWA shell;
- browser WASM/WebGPU constraints;
- optional server-assisted mode with an explicit privacy boundary.

Measure install size, cold start, peak memory, tokens/second, battery/thermal behavior, offline reliability, and update complexity.

### Acceptance gate

- Every advertised backend passes conformance tests.
- Artifacts are immutable and checksum-verified.
- Performance reports identify device, model, context, quantization, and method.
- Experimental model work cannot break the model-free default runtime.

---

# Phase 8 — Product Validation and Release

## Goal

Demonstrate usefulness, transparency, recoverability, privacy, and safety in a limited release.

Stages:

1. maintainer dogfood with disposable identities;
2. invited technical testers;
3. small consenting longitudinal pilot;
4. broader opt-in beta after gates pass.

Release requirements include schema migrations, signed/checksummed artifacts, SBOM, dependency and secret scanning, backup/recovery documentation, reproducible benchmarks, threat model, known limitations, and a vulnerability-reporting process.

### Acceptance gate

- Pilot users can inspect, correct, export, and delete data.
- Reliability and recovery targets are defined and met.
- Prediction and personalization beat named baselines with uncertainty reported.
- No release-blocking privacy or safety findings remain.
- README claims match the evidence report.

---

## Cross-Cutting Test Matrix

| Layer | Required tests |
|---|---|
| Core domain | deterministic unit, property, serialization, and invariant tests |
| State transitions | replay, optimistic-version, rollback, and migration tests |
| Session manager | concurrency, isolation, expiry, reset, deletion |
| Memory | provenance, contradiction, retention, correction, deletion |
| Prediction | pre-outcome issue, delayed resolution, calibration, abstention |
| SANDBOX SIM | immutability, independence, ablation, policy regret |
| Inference | conformance, timeout, cancellation, corruption, OOM |
| API | schema, malformed input, size limits, error codes, rate limits |
| UI | escaping, accessibility, offline and failure states |
| Deployment | clean image, health/readiness, artifact verification |
| Safety | adverse conversations, opt-out, deletion, crisis policies |

---

## Definition of Done for Adaptive Features

An adaptive feature is not complete until:

1. its target behavior is operationally defined;
2. inputs and outputs are typed or schema-validated;
3. state changes are recorded as events;
4. consequential personalization is inspectable and overrideable;
5. a baseline exists;
6. held-out evaluation exists;
7. an ablation measures contribution;
8. calibration and failure rates are reported where confidence is used;
9. privacy and safety effects are reviewed;
10. documentation labels implementation status accurately.

---

## Recommended Pull-Request Sequence

1. **Rust workspace + CI + truthful README.**
2. **Session manager and versioned API.**
3. **SQLite event store and migrations.**
4. **Memory provenance, correction, deletion, export/import.**
5. **Prediction ledger with delayed outcomes.**
6. **Replay benchmark, baselines, and calibration report.**
7. **Independent policy evaluator/SANDBOX SIM.**
8. **Presencing policy, user controls, and adverse-case suite.**
9. **GGUF backend conformance and artifact verification.**
10. **Measured mobile prototypes and architecture decision.**

Each PR must be reviewable, tested, and leave the Rust default path runnable.

---

## Immediate Priority

The next milestone is:

> A model-free Rust server with isolated sessions, typed state, deterministic tests, truthful health reporting, and a fixture that can later prove whether personalization and prediction improve outcomes.

The first model integration comes only after this substrate is reliable.
