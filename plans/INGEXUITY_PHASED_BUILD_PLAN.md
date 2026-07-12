# IngExuity Julia-Everywhere Phased Build Plan

**Implementation line:** `julia-main`  
**Development branches:** `julia-everywhere/*`  
**Updated:** 2026-07-12

## Program objective

Build a robust local-first companion research system in Julia while preserving the features that make the Julia line scientifically interesting:

- multiple-dispatch composition of cognition and policy;
- inspectable Julia-native model operators;
- generic floating-point and integer inference kernels;
- deterministic longitudinal state and prediction replay;
- a single language for model research, evaluation, runtime, and application logic.

The goal is not to prove Julia superior by assumption. The goal is to create experiments capable of showing where Julia helps, where it does not, and whether the custom cognitive and model architecture delivers measurable value.

## Non-negotiable rules

1. A claim requires a fixture, test, benchmark, or study.
2. Producing a response never marks a prediction correct.
3. SANDBOX SIM must not grade a prediction using only the variables that generated it.
4. Randomly initialized model output is not a usable model.
5. Local-first does not mean process-global state.
6. “Julia everywhere” permits C/LLVM-backed Julia packages, but product orchestration and research logic remain Julia-owned.
7. No optimization target may reward dependency, guilt, exclusivity, or social isolation.
8. Mobile and WASM remain research targets until an artifact runs on target hardware.

---

## Phase 0 — Repository truth and executable baseline

### Goals

- Separate Julia and Rust implementation histories.
- Make the Julia package instantiate and test in clean CI.
- Distinguish active, experimental, specified, and unsupported components.
- Exercise the Julia-native transformer without downloading a model.
- Establish deterministic fixtures and package contracts.

### Work

- Maintain `julia-main` as the stable Julia branch.
- Correct `Project.toml` compatibility metadata.
- Add Julia 1.12 CI for package load, tests, and model-free smoke checks.
- Add deterministic tiny-model tests for `NanoGPT.jl`.
- Add tokenizer round-trip tests for supported input domains.
- Record known architectural defects:
  - global conversation state;
  - process-local memory;
  - self-confirming direct-LLM prediction scoring;
  - confidence inflation during retry;
  - non-independent SANDBOX SIM;
  - unvalidated model and mobile claims.
- Require no model download in the default test path.

### Exit gate

- Fresh environment can run `Pkg.instantiate()` and `Pkg.test()`.
- Tiny Julia transformer forward pass is deterministic and finite.
- Tokenizer tests clearly state their supported character domain.
- README and model documentation make no unsupported production claims.
- CI failures identify the exact module or contract that failed.

---

## Phase 1 — Session isolation and typed runtime state

### Goals

Remove `GLOBAL_STATE` as the unit of identity and make every turn an explicit state transition.

### Target interfaces

```julia
abstract type AbstractInferenceBackend end
abstract type AbstractClock end

struct SessionID
    value::UUID
end

struct TurnContext{S,B,C}
    session::S
    backend::B
    clock::C
end

process_turn(state, input, context) -> TurnResult
```

### Work

- Replace singleton state with a session registry keyed by UUID.
- Separate immutable input observations from mutable inferred state.
- Make turn processing transactional:
  - clone or stage state;
  - run inference/policy;
  - validate output;
  - commit only on success.
- Add lifecycle operations: create, inspect metadata, reset, delete, expire.
- Add request IDs and body-size limits.
- Ensure raw message text is not emitted through default logs.
- Define backend health and metadata contracts through multiple dispatch.

### Exit gate

- Concurrent sessions cannot observe or mutate one another.
- Failed inference leaves state unchanged.
- Session reset and deletion have deterministic tests.
- HTTP handlers contain transport logic, not cognitive state logic.

---

## Phase 2 — Durable identity, memory, and provenance

### Goals

Create a durable, correctable user model rather than a bag of strings.

### Memory claim model

Each memory claim must include:

- claim ID;
- session or identity ID;
- normalized proposition;
- claim type;
- source observation IDs;
- confidence and calibration basis;
- created and last-confirmed timestamps;
- validity interval or retention policy;
- sensitivity class;
- status: active, contradicted, superseded, invalidated, deleted;
- optional contradiction/supersession links.

### Work

- Add SQLite.jl with versioned migrations.
- Store append-only observations and events.
- Materialize snapshots for fast startup.
- Add optimistic version checks.
- Add user-visible correction, invalidation, deletion, and export operations.
- Define retention defaults by claim type and sensitivity.
- Add restart, migration, corruption, and deletion tests.
- Prevent raw conversation text from being duplicated unnecessarily in event tables.

### Exit gate

- State survives restart.
- Corrections supersede prior claims without erasing provenance.
- Deletion cascades are verified.
- Storage integrity is part of readiness.
- Identity export/import has a versioned schema and round-trip test.

---

## Phase 3 — Falsifiable prediction ledger

### Goals

Turn prediction into a scientific contract.

### Prediction contract

A prediction records:

- target variable;
- issue time and horizon;
- eligible evidence available at issue time;
- probability distribution or calibrated confidence;
- resolution rule;
- abstention conditions;
- expiry time;
- eventual outcome and scoring method.

### Work

- Separate prediction issuance from intervention choice.
- Resolve predictions only from later observations.
- Add deterministic replay from event logs.
- Evaluate calibration, Brier score, log loss, coverage, and abstention quality.
- Compare against simple baselines:
  - previous-state persistence;
  - majority class;
  - recency heuristic;
  - no-personalization model.
- Remove retry loops that raise confidence without evidence.

### Exit gate

- Every scored prediction can be replayed from recorded eligible evidence.
- No code path self-labels a prediction correct because a response was produced.
- Calibration beats at least one declared baseline on held-out synthetic fixtures before human claims are made.

---

## Phase 4 — Independent SANDBOX SIM and response policy

### Goals

Make SANDBOX SIM an independent policy evaluator rather than a confidence threshold with duplicated inputs.

### Work

- Generate multiple candidate interventions.
- Evaluate candidates with features not identical to the proposal mechanism.
- Include explicit costs:
  - interruption;
  - overconfidence;
  - emotional mismatch;
  - unnecessary personalization;
  - privacy exposure;
  - action risk.
- Add abstain and ask-clarifying-question candidates.
- Use counterfactual replay fixtures.
- Compare:
  - no sandbox;
  - current heuristic sandbox;
  - independent evaluator;
  - randomized candidate selection.

### Exit gate

- Evaluator improves held-out policy utility over no-sandbox and confidence-only baselines.
- Ablations identify which signals provide benefit.
- Failure cases are reported, not hidden by aggregate accuracy.

---

## Phase 5 — Emotional state and presencing

### Goals

Test whether presencing improves interaction quality without manipulating attachment.

### Work

- Separate observed cues from inferred emotional state.
- Represent uncertainty in valence, arousal, stress, and emotional charge.
- Make presencing a candidate policy, not a hard-coded moral truth.
- Add user preference controls for directness, acknowledgment, pacing, and silence.
- Evaluate false-positive presencing and unwanted emotional inference.
- Add crisis-safe behavior boundaries without claiming diagnosis.

### Exit gate

- Presencing improves user-rated appropriateness over matched controls.
- False-positive and false-negative rates are reported.
- Users can inspect and disable emotional adaptation.
- No metric rewards longer engagement for its own sake.

---

## Phase 6 — Julia-native model baseline

### Goals

Produce a trained, reproducible Julia-native baseline before developing novel architecture claims.

### Work

- Repair and specify tokenizer behavior.
- Establish a conventional small transformer control model.
- Add training data manifests and checkpoint provenance.
- Implement training/evaluation scripts in Julia where practical.
- Validate gradients or explicitly separate inference-only kernels.
- Add numerical tests for layer normalization, masking, sampling, and weight loading.
- Benchmark against a similarly sized external local model.

### Required metrics

- held-out cross entropy/perplexity;
- generation quality on fixed prompts;
- tokens per second;
- first-token latency;
- peak resident memory;
- checkpoint size;
- deterministic reproducibility where configured;
- numerical error against reference implementations.

### Exit gate

- A documented checkpoint can be reproduced or independently verified.
- Results beat random and trivial baselines.
- The model is never selected as the default backend merely because it is Julia-native.

---

## Phase 7 — Linguist-LSA research program

### Goals

Test the proposed selective-state-space and linear-merge architecture against the conventional transformer baseline.

### Research tracks

1. **Selective SSM memory**
   - long-context synthetic recall;
   - state stability;
   - update complexity;
   - reset and contamination behavior.

2. **Linear-merge attention**
   - approximation error against causal attention;
   - constant-memory claim verification;
   - latency crossover points;
   - degradation on retrieval and induction tasks.

3. **Low-rank SwiGLU**
   - parameter and activation savings;
   - quality loss by rank;
   - rank adaptation by layer.

4. **Integer-oriented inference**
   - explicit quantization scales and zero points;
   - saturation behavior;
   - SIMD kernel benchmarks;
   - accuracy loss by operator and bit width.

### Exit gate

- Each novel operator has a reference implementation, optimized implementation, and ablation.
- Memory and speed claims are measured on target hardware.
- Quality is compared at matched parameter count and matched compute.
- Negative results remain in the research record.

---

## Phase 8 — Deployment and device feasibility

### Goals

Package the Julia runtime responsibly and determine which device targets are real.

### Work

- Build reproducible native application images.
- Measure cold start, sysimage size, and package latency.
- Test CPU architectures separately.
- Prototype mobile embedding before promising WASM.
- Evaluate whether selected kernels require C/LLVM extensions while retaining Julia ownership of orchestration.
- Add signed model manifests and checksums.
- Add backup, recovery, and storage-quota behavior.

### Exit gate

- A release artifact boots and passes health checks on declared hardware.
- Model and database paths are configurable.
- No runtime download is unpinned or unverified.
- Mobile claims name an actual tested device and artifact.

---

## Phase 9 — Longitudinal product research

### Goals

Determine whether IngExuity provides durable value without manipulative attachment design.

### Study design

- staged synthetic and internal tests;
- consenting short human sessions;
- longitudinal opt-in pilot;
- clear data deletion and export;
- comparison against non-personalized and memory-only baselines.

### Primary outcomes

- task success;
- correction burden;
- prediction calibration;
- user-rated appropriateness;
- unwanted personalization rate;
- memory intrusion rate;
- trust calibration;
- retention only as a secondary descriptive measure.

### Prohibited success criteria

- guilt about turning the system off;
- exclusivity from human relationships;
- increased dependence;
- conversation length without demonstrated benefit.

---

## Immediate PR sequence

1. `julia-everywhere/phase-0`: repository truth, package metadata, Julia CI, native-model smoke tests.
2. `julia-everywhere/session-runtime`: explicit sessions and transactional turns.
3. `julia-everywhere/persistence`: SQLite events, snapshots, provenance, correction/deletion.
4. `julia-everywhere/prediction-ledger`: issue/resolve/replay/scoring contracts.
5. `julia-everywhere/sandbox-policy`: independent candidate evaluation.
6. `julia-everywhere/model-baseline`: trained Julia transformer control.
7. `julia-everywhere/linguist-lsa`: novel operator experiments behind benchmarks.

## Definition of a robust Julia-everywhere release

A release is robust only when:

- clean installation and tests pass;
- state is isolated and durable;
- predictions are externally resolvable and replayable;
- user model claims are inspectable, correctable, and deletable;
- inference backends have conformance tests;
- model artifacts have provenance and checksums;
- claimed performance is measured;
- safety and privacy boundaries are explicit;
- unsupported research targets are labeled as such.
