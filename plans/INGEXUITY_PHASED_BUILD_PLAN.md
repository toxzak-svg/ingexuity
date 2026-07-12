# IngExuity — Robust Phased Build Plan

**Status:** Engineering baseline and execution roadmap  
**Updated:** 2026-07-12  
**Scope:** Turn the current prototype into a reliable, measurable, privacy-preserving local companion system.  
**Product thesis:** IngExuity should become useful through accumulated interaction and explicit user modeling, not through claims that cannot be measured.

---

## 1. What This Plan Changes

The earlier roadmap mixed implemented behavior, desired architecture, model research, deployment targets, and product claims into a single timeline. This plan replaces schedule-based promises with **evidence-gated phases**. A phase is complete only when its acceptance gate passes.

The current repository contains a meaningful prototype, but the implementation and documentation have drifted:

- The active runtime uses `HTTP.jl`, `JSON.jl`, and a `LlamaCpp` GGUF backend.
- `NanoGPT.jl`, `IGSDCore.jl`, and `TrainedModel.jl` are present as experimental or disabled paths, not validated production backends.
- Conversation and prediction state are process-global.
- Memory is currently an in-memory vector with a default 24-hour validity window.
- The module pipeline exists, but the direct local-model path bypasses most of it.
- A nonempty direct-model response is currently recorded as a correct prediction, so the reported accuracy is not yet a valid measurement.
- SANDBOX SIM primarily rechecks the same heuristic state used to generate predictions; it is not yet an independent counterfactual evaluator.
- Module tests are substantial, but the repository lacks a complete production test matrix for persistence, concurrency, API contracts, deployment, security, and model failure modes.

This is not a reason to discard the project. It identifies the correct starting point.

---

## 2. Non-Negotiable Engineering Principles

1. **Repository truth before product narrative.** Documentation must distinguish active, experimental, planned, and deprecated components.
2. **No self-confirming metrics.** Output generation cannot label its own prediction as correct merely because output was produced.
3. **Session isolation by construction.** One user's state, memory, or inference context must never appear in another session.
4. **Local-first does not mean security-free.** Local data still needs access control, deletion, migration, integrity checks, and a threat model.
5. **Uncertainty is a valid result.** The system must be able to abstain, ask, or defer rather than inflate confidence.
6. **Every adaptive mechanism needs a baseline and an ablation.** A module is useful only if removing it measurably worsens an appropriate outcome.
7. **Safety is a system property.** Emotional interaction, memory, prediction, and response policy require explicit constraints and adverse-case tests.
8. **Portability follows stable state semantics.** Multi-device identity cannot be reliable until state schemas, migrations, conflict rules, and provenance are defined.
9. **Model architecture is replaceable infrastructure.** IngExuity's product value must not depend on an unvalidated custom transformer.
10. **Claims are earned at acceptance gates.** README language must not outrun reproducible evidence.

---

## 3. Target Runtime Architecture

```text
Client
  -> API boundary
     -> authenticated/authorized Session ID
        -> Session Manager
           -> Conversation State
           -> User Model
           -> Memory Store
           -> Prediction Ledger
           -> Event Log
        -> Turn Orchestrator
           -> Comprehension / feature extraction
           -> Prediction proposals
           -> Candidate response policies
           -> Independent policy checks
           -> Inference backend
           -> Output renderer
        -> Outcome Resolver
           -> delayed labels
           -> calibration update
           -> user-model update
           -> audit event
```

### Required interfaces

- `InferenceBackend`: load, health, generate, cancel, unload, model metadata.
- `StateStore`: create session, load snapshot, append event, transact, migrate, delete, export.
- `MemoryStore`: write, retrieve, invalidate, search, explain provenance, enforce retention.
- `PredictionLedger`: issue prediction, resolve outcome, expire unresolved prediction, score resolved set.
- `PolicyEvaluator`: assess response candidates without modifying the state it judges.
- `Clock` and `IDGenerator`: injectable for deterministic replay and testing.

The initial implementation can remain Julia. The interfaces matter more than premature service decomposition.

---

# Phase 0 — Establish Repository Truth and a Reproducible Baseline

## Goal

Make the current system build, test, run, and describe itself consistently before adding new cognition claims.

## Work

### 0.1 Reconcile documentation and active code

- Mark each backend and deployment target as `active`, `experimental`, `planned`, or `deprecated`.
- Align the README, architecture document, `Project.toml`, Dockerfile, and source comments.
- Correct package naming drift between `IngExuity` and `IngEnuity`.
- Document the actual active conversation path and the separate heuristic module-pipeline fallback.
- Move speculative performance and mobile claims into research documents until demonstrated.

### 0.2 Normalize the Julia project

- Put Julia compatibility under `[compat]` and declare compatibility for every direct dependency.
- Remove unused dependencies or place experimental backends behind explicit optional environments.
- Commit a reproducible `Manifest.toml` for the application deployment path.
- Pin model artifact URLs by immutable revision and verify SHA-256 before loading.
- Remove build steps that suppress package-load failure with `|| true`.

### 0.3 Add continuous integration

CI must run at minimum:

1. clean instantiate;
2. package load;
3. unit tests;
4. formatting/static checks selected for the project;
5. server smoke test;
6. invalid-request API tests;
7. a model-free fallback test;
8. Docker build verification without downloading mutable artifacts during the test.

### 0.4 Define baseline behavior

Create a checked-in, privacy-safe evaluation fixture containing deterministic synthetic conversations for:

- ordinary information seeking;
- short and ambiguous messages;
- topic changes;
- stress-language false positives;
- emotional disclosure;
- multi-turn preference correction;
- two concurrent users with conflicting preferences;
- model unavailable and timeout behavior.

## Acceptance gate

- A clean checkout can instantiate, load, and pass tests in CI.
- The HTTP server passes health, chat, malformed JSON, missing field, and fallback tests.
- Documentation correctly labels every backend and deployment target.
- No build step hides a failed package load.
- Baseline evaluation results are saved as versioned artifacts.

---

# Phase 1 — Isolate Sessions and Harden the Runtime

## Goal

Make the application correct under multiple users, multiple requests, failures, and hostile input.

## Work

### 1.1 Replace process-global conversation state

- Introduce a `SessionManager` keyed by an opaque UUID or equivalent unguessable token.
- Keep mutable state inside a session boundary.
- Define session creation, lookup, expiration, deletion, and reset semantics.
- Protect state transitions with a per-session lock or actor-style turn queue.
- Make module functions accept explicit state rather than reading hidden globals.

### 1.2 Define an API contract

- Version endpoints under `/api/v1/`.
- Define request/response schemas, stable error codes, maximum body sizes, and content-type checks.
- Carry a session identifier explicitly.
- Add request IDs and structured logs without logging raw private conversation content by default.
- Add cancellation, inference timeout, and bounded queue behavior.
- Return truthful health/readiness information: server alive, model loaded, storage writable.

### 1.3 Secure output and inputs

- Remove direct `innerHTML` insertion of user and model text; render through text nodes or sanitization.
- Add rate limits suitable for local and hosted modes.
- Define CORS and host-binding behavior explicitly.
- Prevent path injection in model loading and identity import.
- Reject oversized or malformed identity bundles before parsing.

### 1.4 Control model lifecycle

- Load the model once through `InferenceBackend`.
- Report exact model file, quantization, context size, runtime, checksum, and load state.
- Bound generation length and concurrency.
- Test out-of-memory, missing model, corrupt model, cancellation, and timeout paths.

## Acceptance gate

- A concurrency test proves that two sessions cannot observe or mutate each other's state.
- Thread/race testing produces no corrupted turn counts, prediction ledgers, or memory records.
- User/model output cannot execute HTML or script in the embedded UI.
- All API errors are machine-readable and do not expose stack traces or private state.
- Model failure degrades to an explicit unavailable/fallback state rather than silent false success.

---

# Phase 2 — Durable Memory, Identity, and User Control

## Goal

Create a persistent, inspectable identity substrate with explicit provenance and lifecycle rules.

## Work

### 2.1 Introduce a transactional store

Use SQLite first unless measurements justify another database. Define migrations for:

- sessions;
- turns and events;
- memory claims;
- observations/evidence;
- user-model features;
- predictions and outcomes;
- model/backend metadata;
- consent and retention settings.

Do not serialize arbitrary Julia objects as the long-term public format.

### 2.2 Replace raw string memories with claims

A memory record should include:

- stable ID;
- subject and predicate or typed claim category;
- normalized value plus display value;
- `observed_at`, `valid_from`, `valid_until`, and `invalidated_at`;
- source event IDs;
- confidence and uncertainty reason;
- sensitivity class;
- retention policy;
- contradiction/supersession links.

The system must distinguish an observation from an inference and a user-confirmed fact.

### 2.3 Build identity export/import

- Define a versioned, documented identity bundle schema.
- Include checksums and a manifest of included data.
- Support full export, selective export, and user-readable inspection.
- Encrypt exported bundles with a user-controlled secret or platform keystore integration.
- Validate schema version and run migrations on import.
- Define merge/conflict behavior before multi-device synchronization.

### 2.4 Add data rights to the product

The user must be able to:

- inspect why a memory exists;
- correct it;
- invalidate it;
- delete a session or all data;
- set retention windows;
- disable sensitive inference categories;
- export a portable copy.

## Acceptance gate

- State survives restart without cross-session leakage.
- Database migrations are tested forward from every released schema.
- Corruption and interrupted-write recovery are tested.
- Export -> delete -> import produces an equivalent identity under a canonical comparison.
- Deletion and correction remove the affected claim from future retrieval and prediction.
- Every retrieved memory can identify its provenance.

---

# Phase 3 — Make Prediction Falsifiable

## Goal

Turn “prediction-first” from a narrative into a measurable subsystem.

## Work

### 3.1 Define prediction contracts

Every prediction must specify:

- prediction ID and session ID;
- target variable;
- target horizon or resolution deadline;
- finite outcome space or scoring rule;
- confidence/probability distribution;
- evidence available at issue time;
- model/policy version;
- abstention state;
- resolution source;
- final outcome and resolution timestamp.

Examples of valid targets:

- whether the next user turn continues the current topic;
- whether the next turn is a question, correction, elaboration, or topic shift;
- preferred response depth among a defined set;
- whether acknowledgment-first or solution-first is selected by explicit user feedback;
- whether a previously stated preference remains valid after a defined horizon.

“User needs support” is not scoreable until its outcome and label source are operationally defined.

### 3.2 Separate issue time from resolution time

- Store predictions before generating the response.
- Resolve only from later observable evidence, explicit user feedback, or blinded annotation.
- Expire unresolved predictions rather than marking them correct.
- Never infer correctness from response non-emptiness, user continuation alone, or the predictor's own explanation.

### 3.3 Establish baselines

Compare IngExuity against:

- majority-class prediction;
- last-topic persistence;
- recency-only user preference;
- generic non-personalized response policy;
- underlying model without IngExuity state;
- oracle upper bounds where labels permit them.

### 3.4 Use proper metrics

Report at minimum:

- Brier score or multiclass log loss;
- calibration error and reliability plots;
- accuracy/F1 only where class balance makes them meaningful;
- coverage at each abstention threshold;
- regret or utility for response-policy choices;
- per-user and macro-averaged results;
- confidence intervals and sample counts;
- unresolved/expired prediction rate.

### 3.5 Build deterministic replay

The replay harness must reconstruct a turn using:

- the exact prior event stream;
- model/backend version;
- prompt/policy version;
- random seed where applicable;
- prediction candidates;
- selected action;
- later outcome label.

## Acceptance gate

- No metric is updated without a prediction issued before its outcome.
- A held-out replay set compares at least one personalized predictor to named baselines.
- Calibration, coverage, and sample size are reported alongside accuracy.
- The system can abstain without being counted as correct.
- Reported improvement survives an ablation that removes user-specific state.

---

# Phase 4 — Build a Real User Model

## Goal

Learn durable, correctable user-specific patterns without treating every interaction as proof.

## Work

### 4.1 Separate feature classes

Maintain distinct stores for:

- explicit preferences stated by the user;
- observed behavior;
- inferred preferences;
- temporary conversational state;
- long-horizon routines;
- communication-style estimates;
- emotional-context estimates.

Each class needs different retention, confidence, and confirmation rules.

### 4.2 Use evidence-weighted online updating

- Stop increasing global confidence merely because another turn occurred.
- Update confidence based on resolved outcomes and source reliability.
- Decay or invalidate stale patterns.
- Preserve contradictions instead of silently overwriting them.
- Use hierarchical or Bayesian shrinkage so sparse user evidence does not produce extreme confidence.

### 4.3 Add retrieval and decision provenance

For each personalization decision, retain:

- which memories/features were considered;
- which were excluded and why;
- how uncertainty affected the decision;
- whether a generic fallback was available;
- the final policy choice.

### 4.4 Solve cold start without fake familiarity

- Start with a transparent generic interaction policy.
- Ask high-information, low-burden preference questions only when useful.
- Use reversible defaults.
- Demonstrate early recognition by recalling confirmed facts, not by pretending deep understanding.

## Acceptance gate

- A correction test shows that explicit correction overrides a weaker inference.
- Confidence does not monotonically increase with turn count.
- Personalization beats a non-personalized baseline on held-out users or sessions.
- False personalization and stale-memory rates are measured.
- The user can inspect and alter the evidence behind a personalization choice.

---

# Phase 5 — Rebuild SANDBOX SIM as Independent Policy Evaluation

## Goal

Use simulation to choose among candidate response policies, not to restate the generator's heuristics.

## Work

### 5.1 Define what is simulated

SANDBOX SIM should evaluate structured candidate policies such as:

- acknowledgment-first vs. solution-first;
- concise vs. detailed;
- ask-one-question vs. answer-now;
- direct vs. hedged;
- retrieve-memory vs. avoid-memory;
- act vs. request confirmation.

It should not claim to simulate a whole human mind.

### 5.2 Enforce evaluator independence

- Candidate generation and evaluation must be separate interfaces.
- The evaluator receives an immutable state snapshot.
- It cannot increase prediction confidence or mutate the user model while judging.
- Evaluation features and weights are versioned.
- Where the same base model is used for proposal and critique, report that dependence explicitly.

### 5.3 Learn from resolved policy outcomes

- Randomize among safe candidates during an opt-in exploration phase.
- Use contextual bandit or off-policy evaluation methods only when assumptions are documented.
- Track user-specific utility and global safety constraints separately.
- Penalize unnecessary interruption, repetition, manipulation, and overconfident memory use.

### 5.4 Run ablations

Compare:

1. no sandbox;
2. threshold-only filter;
3. independent rule evaluator;
4. learned evaluator;
5. evaluator plus user-specific features.

## Acceptance gate

- SANDBOX SIM improves a preregistered response-policy metric over no-sandbox and threshold baselines.
- The evaluator cannot mutate the state it evaluates.
- Retry does not raise confidence without new evidence.
- Failure cases and negative results are retained, not hidden.
- Safety constraints dominate learned utility when they conflict.

---

# Phase 6 — Emotional Interaction and Safety

## Goal

Make “staying present” a calibrated, user-controllable interaction policy rather than a brittle keyword threshold.

## Work

### 6.1 Replace fixed emotional claims with uncertainty-aware estimates

- Treat stress, valence, arousal, and emotional charge as uncertain observations.
- Measure false positives and false negatives across different writing styles.
- Do not infer a stable mental-health condition from conversational language.
- Prefer clarification when confidence is low and the cost of a wrong response is high.

### 6.2 Operationalize presencing

Define selectable policies:

- acknowledge and continue solving;
- acknowledge and ask whether the user wants listening or solutions;
- stay with the topic for one turn;
- provide immediate concrete help;
- use a user-configured default.

Avoid a hardcoded early return that blocks useful help whenever a threshold fires.

### 6.3 Add safety boundaries

- Crisis and self-harm scenarios require a dedicated, tested response policy.
- Do not encourage exclusivity, dependency, guilt, secrecy, or replacement of human relationships.
- Do not use emotional vulnerability to increase retention.
- Avoid anthropomorphic claims the system cannot support.
- Make memory use visible when it affects sensitive responses.

The former “guilt at deleting” or “irreplaceable” measures must not be optimization targets. Product value should be measured through usefulness, trust, correction success, continuity, and voluntary retention without coercive attachment.

### 6.4 Build an adverse-case suite

Include:

- sarcasm and profanity;
- terse technical users;
- grief and anger;
- repeated corrections;
- contradictory preferences;
- manipulation attempts;
- delusional framing;
- crisis language;
- user asks the system to stop personalizing;
- user asks for memory deletion.

## Acceptance gate

- Presencing is compared against at least one acknowledgment-plus-help baseline.
- False-positive emotional intervention rate is reported.
- Users can configure or disable emotional adaptation.
- Safety scenarios pass deterministic policy checks and human review criteria.
- Retention metrics are not based on guilt, distress, or exclusivity.

---

# Phase 7 — Inference Backends, Performance, and Edge Deployment

## Goal

Support practical local inference without tying the product roadmap to an unproven model stack.

## Work

### 7.1 Stabilize the GGUF baseline

- Treat the active LlamaCpp path as the reference backend.
- Record model license, source revision, checksum, quantization, context, and runtime settings.
- Add reproducible latency, memory, throughput, and quality benchmarks.
- Test representative low-resource hardware rather than relying on estimated specifications.

### 7.2 Make backends pluggable

Implement the same conformance suite for:

- GGUF/llama.cpp;
- any trained adapter export;
- IGSDCore experiments;
- NanoGPT or future Julia-native models;
- remote development backends, if ever enabled, behind an explicit privacy boundary.

A backend is not production-ready until it passes generation, cancellation, context, determinism, metadata, and failure tests.

### 7.3 Run a mobile feasibility spike before committing

Compare at least:

- native llama.cpp through Android NDK/JNI;
- a small native Julia application where supported;
- browser/WebGPU or WASM constraints;
- local companion service plus PWA shell;
- server-assisted mode as an optional, explicit privacy tradeoff.

Measure install size, cold start, peak RAM, tokens/second, battery/thermal behavior, offline reliability, and update complexity.

### 7.4 Keep custom-model research separate

A Julia-native transformer is a research track until it demonstrates a useful quality/latency/privacy advantage over the GGUF baseline. It must not block the memory, prediction, safety, or product work.

## Acceptance gate

- Backend conformance tests pass for every advertised backend.
- Model artifacts are immutable and checksum-verified.
- Performance claims include device, model, context, quantization, and measurement method.
- A mobile architecture decision is supported by measured prototypes.
- The default experience remains functional when experimental backends are absent.

---

# Phase 8 — Product Validation and Release Readiness

## Goal

Demonstrate that IngExuity is useful, understandable, recoverable, and safe enough for a limited release.

## Work

### 8.1 Build transparent onboarding

- Explain what is local, what is stored, and what is inferred.
- Let the user select memory and personalization defaults.
- Show how to inspect, correct, export, and delete data.
- Avoid claiming familiarity before evidence exists.

### 8.2 Run a staged pilot

1. maintainer dogfood with synthetic and disposable identities;
2. invited technical testers;
3. small consenting longitudinal pilot;
4. broader opt-in beta only after privacy and safety gates pass.

Collect explicit qualitative feedback and instrumented metrics without collecting raw private text by default.

### 8.3 Release engineering

- versioned schemas and migrations;
- signed or checksummed artifacts;
- software bill of materials;
- dependency and secret scanning;
- backup/recovery documentation;
- reproducible benchmark report;
- threat model and privacy documentation;
- known-limitations document;
- incident and vulnerability reporting process.

## Acceptance gate

- Pilot users can successfully inspect, correct, export, and delete their data.
- Crash-free and recovery targets are defined and met for the pilot.
- Prediction and personalization results beat named baselines with uncertainty reported.
- Safety and privacy reviews have no unresolved release-blocking findings.
- README claims match the evidence report and known limitations.

---

## 4. Cross-Cutting Test Matrix

| Layer | Required tests |
|---|---|
| Pure modules | deterministic unit and property tests |
| State transitions | invariant, replay, rollback, and migration tests |
| Session manager | concurrency, isolation, expiry, deletion |
| Memory | provenance, contradiction, retention, correction, deletion |
| Prediction | pre-outcome issuance, delayed resolution, calibration, abstention |
| SANDBOX SIM | immutability, independence, ablation, policy regret |
| Inference | conformance, timeout, cancellation, corruption, OOM |
| API | schema, malformed input, size limits, error codes, rate limits |
| UI | escaping, accessibility, offline/failure states |
| Deployment | clean image build, health/readiness, artifact verification |
| Safety | adverse conversations, personalization opt-out, crisis policies |

---

## 5. Definition of Done for Any Adaptive Feature

An adaptive feature is not done until all are true:

1. The target behavior is operationally defined.
2. The feature's inputs and outputs are typed or schema-validated.
3. State changes are recorded in the event log.
4. The user can inspect or override consequential personalization.
5. A baseline exists.
6. A held-out evaluation exists.
7. An ablation shows whether the feature adds value.
8. Calibration and failure rates are reported where confidence is used.
9. Privacy and safety effects are reviewed.
10. Documentation says what is implemented and what remains experimental.

---

## 6. Recommended Pull-Request Sequence

1. **Repository truth + CI:** dependencies, docs status, clean build, smoke tests.
2. **Session isolation:** remove global state, add session API and concurrency tests.
3. **Storage foundation:** SQLite migrations, event log, memory provenance.
4. **Identity controls:** export/import, correction, deletion, retention settings.
5. **Prediction ledger:** typed targets, delayed outcomes, proper scoring.
6. **Replay benchmark:** baselines, calibration report, ablation harness.
7. **Policy-based SANDBOX SIM:** immutable candidate evaluation and experiments.
8. **Presencing policy:** user controls, adverse-case tests, acknowledgment-plus-help baseline.
9. **Backend interface:** GGUF conformance, artifact verification, reproducible benchmarks.
10. **Mobile feasibility:** measured prototypes and architecture decision record.

Each pull request should be small enough to review, contain tests, and leave `main` runnable.

---

## 7. Immediate Priority

The next implementation milestone is **not** a larger model or more modules. It is:

> A reproducible server with isolated sessions, durable state, truthful metrics, and a replay harness that can prove whether personalization and prediction improve outcomes.

Once that exists, IngExuity can become a serious product and a credible research program rather than an accumulation of plausible-sounding mechanisms.
