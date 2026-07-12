# IngExuity — Expanded Research Directions

**Status:** Research agenda  
**Updated:** 2026-07-12  
**Purpose:** Define the scientific questions behind a prediction-first, local, longitudinal companion system and specify experiments that can disprove as well as support its claims.

---

## 1. Research Position

IngExuity's strongest idea is not that a companion needs more personality. It is that a useful long-term companion may require a **continually updated belief state about one user**, explicit predictions about near-future needs and behavior, memory with temporal validity, and a response policy selected from those beliefs.

That idea is scientifically interesting only if it is separated into falsifiable components.

The central research question is:

> Does explicit, longitudinal user modeling with calibrated prediction and policy selection improve user outcomes beyond a capable language model with ordinary conversational context and memory retrieval?

The default null hypothesis is:

> Most apparent benefit comes from the base model, recent context, generic empathy patterns, or increased prompt length; the named IngExuity modules add no reliable value once those controls are included.

The research program should be designed to reject or retain that null hypothesis honestly.

---

## 2. Claims That Must Be Separated

The project currently brings several distinct claims together. They need different evidence.

| Claim | Required evidence |
|---|---|
| The system predicts the user's next state | pre-registered prediction targets, delayed labels, proper scoring rules |
| Personalization improves responses | randomized or counterbalanced comparison with a non-personalized baseline |
| Memory improves continuity | retrieval and response ablations with stale-memory and false-memory measurements |
| SANDBOX SIM improves policy choice | no-sandbox baseline, independent evaluator, regret or utility comparison |
| Presencing improves emotional interactions | controlled comparison against acknowledgment-plus-help and user-choice policies |
| Identity persists across devices | state-equivalence, migration, conflict, and user-recognition studies |
| Hardware creates useful “personality texture” | blinded comparisons showing consistent, preferred differences not caused by defects |
| A Julia-native model is advantageous | quality, latency, memory, power, deployment, and maintainability comparison |
| The system becomes more useful over time | longitudinal outcome curves with attrition and novelty effects controlled |
| The companion is safe to rely on | adverse-case testing, user control, non-manipulation, recovery, and escalation behavior |

No single satisfaction survey can validate all of these.

---

## 3. Core Research Infrastructure

Before studying advanced mechanisms, build a shared experimental substrate.

### 3.1 Versioned event stream

Every experimental turn should record, with privacy-preserving controls:

- anonymized participant/session identifier;
- event and turn IDs;
- timestamp and logical sequence number;
- model/backend and policy versions;
- state snapshot hash;
- memories retrieved and their provenance IDs;
- predictions issued before output;
- candidate policies considered;
- evaluator scores;
- selected policy;
- user-visible response;
- later outcome/label and label source;
- correction, deletion, or consent events.

Raw text should be optional, minimized, encrypted, access-controlled, and excluded from routine telemetry where derived labels are sufficient.

### 3.2 Deterministic replay

A replay system must reproduce policy decisions from a frozen event prefix. Where model sampling prevents byte-identical generation, store candidate outputs and separately replay the deterministic selection path.

### 3.3 Evaluation registry

Every experiment should register:

- research question;
- primary and secondary hypotheses;
- unit of analysis;
- inclusion/exclusion rules;
- train/development/test split policy;
- primary metric;
- stopping rule;
- correction for multiple comparisons;
- known confounders;
- safety constraints;
- conditions under which the hypothesis is considered unsupported.

### 3.4 Baseline ladder

Use progressively stronger controls:

1. generic fixed response;
2. base model, current turn only;
3. base model plus recent conversation window;
4. base model plus generic retrieval;
5. base model plus user-specific memory;
6. user-specific memory plus explicit predictor;
7. predictor plus policy evaluator/SANDBOX SIM;
8. full IngExuity system.

A component earns its place only if it improves over the immediately weaker baseline or provides another measurable benefit such as lower compute, better privacy, or improved interpretability.

---

# Research Direction 1 — Operationalizing “What the User Needs”

## Problem

“Need” is too broad and subjective to function as a prediction target. It can be relabeled after the fact to match almost any response.

## Questions

- Which user-relevant states can be defined before observing the next turn?
- Which labels can be obtained with low burden and acceptable reliability?
- At what horizons is prediction useful: next turn, next session, next day, or longer?
- Does predicting a need improve policy selection, or is direct policy learning sufficient?

## Candidate target ontology

### Observable next-turn targets

- continue topic / change topic;
- ask question / correct / elaborate / acknowledge / disengage;
- short / medium / long expected response preference;
- explicit request for information / action / emotional acknowledgment;
- whether the user rejects an invoked memory;
- whether clarification is required.

### User-reported policy targets

- wanted listening, advice, factual answer, action, or brainstorming;
- preferred directness level;
- preferred response depth;
- whether memory use felt helpful, irrelevant, or intrusive;
- whether the response reduced effort or created more work.

### Longitudinal targets

- preference persistence;
- recurring task/routine occurrence;
- return to an unresolved project;
- correction or invalidation of a stored belief;
- whether a proactive suggestion is accepted when explicitly permitted.

## Experiments

1. Build an annotation study comparing user self-labels, independent annotators, and model-generated labels.
2. Measure inter-rater agreement and label stability over time.
3. Compare a target-specific predictor with a generic “need” classifier.
4. Evaluate whether a correct target prediction improves downstream response utility.

## Metrics

- agreement coefficients appropriate to the label type;
- label missingness and user burden;
- Brier score/log loss;
- calibration by target and user;
- downstream policy utility conditional on prediction correctness.

## Falsification condition

If operational targets have low agreement, high user burden, or no downstream utility, “need prediction” should be narrowed or replaced with direct policy preference modeling.

---

# Research Direction 2 — Multi-Horizon User Prediction

## Problem

Near-term conversational prediction and long-term trajectory modeling are different tasks. Combining them into “precognition” risks hiding poor performance.

## Questions

- How does predictability change across horizons?
- Which features matter at each horizon?
- Do personalized predictors beat population and recency baselines?
- Can the system remain calibrated as user behavior changes?

## Proposed model

Represent predictions at explicit horizons:

- `H0`: current-turn latent policy preference;
- `H1`: next-turn action;
- `H2`: remainder-of-session state;
- `H3`: next-session topic or goal;
- `H4`: recurring routine or preference validity over days/weeks.

Use separate heads or models and never average them into one undifferentiated accuracy number.

## Experiments

- Compare majority, recency, n-gram/state-transition, logistic, tree-based, Bayesian, sequence-model, and base-LLM predictors.
- Evaluate global, per-user, and hierarchical models.
- Test online change-point detection when preferences shift.
- Measure the value of abstention under sparse history.
- Run leave-future-out splits; never randomly mix future turns into training for earlier predictions.

## Metrics

- log loss/Brier score by horizon;
- expected calibration error and reliability diagrams;
- top-k accuracy where outcome spaces are large;
- coverage-risk curves;
- adaptation lag after a preference shift;
- per-user variance and worst-decile performance.

## Falsification condition

If explicit personalized prediction does not outperform recency/context baselines on held-out future data, it should not be marketed as a core capability at that horizon.

---

# Research Direction 3 — User Modeling as Belief Revision

## Problem

Current-style profile accumulation can turn transient behavior into permanent identity. A robust user model must distinguish evidence classes, uncertainty, contradictions, and change.

## Questions

- What representation supports correction and temporal validity without becoming too rigid?
- How should explicit statements, observed behavior, and inferred preferences be weighted?
- How quickly should beliefs decay or adapt?
- Can the model explain why it believes something about the user?

## Candidate approaches

- typed claim graph with provenance;
- Bayesian belief updates;
- truth-maintenance systems with contradiction links;
- temporal knowledge graphs;
- event-sourced materialized views;
- hierarchical user/population priors;
- change-point and concept-drift detection.

## Experiments

1. **Correction benchmark:** seed a weak inference, then provide an explicit correction; measure turns until behavior changes.
2. **Contradiction benchmark:** present context-dependent preferences and test whether the system avoids flattening them.
3. **Staleness benchmark:** change a preference after a long history and measure adaptation lag.
4. **Provenance benchmark:** ask the system to explain the evidence for a belief; score faithfulness against logged provenance.
5. **Sparse-user benchmark:** compare per-user learning with hierarchical shrinkage under little data.

## Metrics

- correction success and latency;
- stale-belief invocation rate;
- contradiction preservation;
- false personalization rate;
- provenance precision/recall;
- calibration of belief confidence;
- user-rated controllability.

## Falsification condition

If a simpler explicit preference store performs as well as a richer inferred user model, use the simpler system until richer modeling demonstrates value.

---

# Research Direction 4 — Active Learning Without Interrogation

## Problem

The system can reduce uncertainty by asking questions, but excessive questions create friction and can feel invasive.

## Questions

- When is asking worth the interruption?
- Which question maximizes information gain relevant to the current decision?
- Can uncertainty be reduced through passive observation without overinterpreting behavior?
- How should the user control sensitive categories?

## Candidate formulation

Treat clarification as a decision with costs:

```text
expected value of asking
  = expected improvement in response utility
  - interruption cost
  - privacy/sensitivity cost
  - delay cost
```

## Experiments

- Compare always-ask, never-ask, threshold, information-gain, and user-configured policies.
- Randomize safe low-stakes clarification decisions.
- Measure repeated-question annoyance and preference retention.
- Test whether asking about sensitive domains is appropriately suppressed without explicit consent.

## Metrics

- task utility after clarification;
- questions per session;
- unnecessary-question rate;
- information gain per question;
- abandonment or topic-change after a question;
- user-rated intrusiveness.

## Falsification condition

If active questioning adds little utility or materially increases friction, prefer reversible defaults and explicit settings.

---

# Research Direction 5 — SANDBOX SIM and Counterfactual Policy Evaluation

## Problem

A simulator that uses the same heuristics as the proposer can create an illusion of validation. The key question is whether offline candidate evaluation predicts real user outcomes.

## Questions

- What exactly is being simulated: user response, policy utility, safety risk, or task completion?
- Can evaluator scores predict observed outcomes out of sample?
- How much independence from the response generator is necessary?
- When does simulation outperform a simple rule or direct policy model?

## Candidate architecture

```text
state snapshot
  -> candidate policy generator
  -> candidate responses or response plans
  -> independent evaluators
       * user-preference utility
       * task utility
       * uncertainty
       * safety/constraint checks
  -> selector
  -> observed outcome
  -> evaluator calibration update
```

## Experimental ladder

1. fixed rule selector;
2. confidence threshold;
3. hand-coded independent evaluator;
4. learned outcome model;
5. LLM critic without user state;
6. LLM critic with user state;
7. ensemble with hard safety constraints.

Use logged bandit evaluation cautiously; document overlap and ignorability assumptions. Prefer prospective randomization among policies already judged safe.

## Metrics

- predicted vs. observed utility correlation;
- calibration of evaluator scores;
- policy regret;
- selection accuracy where a best policy can be labeled;
- safety violation rate;
- compute and latency cost;
- performance under evaluator ablation.

## Falsification condition

If a learned simulator cannot beat a simple selector on prospective outcomes, keep SANDBOX SIM as a constrained rules engine rather than claiming counterfactual understanding.

---

# Research Direction 6 — Presencing and Emotional Interaction Policy

## Problem

Acknowledging emotion before problem-solving may help, but fixed lexical thresholds can misread terse, technical, sarcastic, culturally different, or profanity-heavy users. A hard early return can also withhold requested help.

## Questions

- When does acknowledgment-first improve outcomes?
- Is user choice (“listen or solve?”) better than inference?
- Does acknowledgment plus immediate useful help outperform acknowledgment-only?
- How should the system adapt to users who dislike emotional framing?

## Candidate policies

- direct answer;
- acknowledgment then answer;
- acknowledgment then one preference question;
- one-turn listening/presencing;
- user-configured default;
- abstain from emotional interpretation and ask a neutral clarification.

## Experiments

- Within-user counterbalanced comparisons for low-risk scenarios.
- Stress-test false positives using technical frustration, slang, profanity, sarcasm, and short messages.
- Measure effects separately for task satisfaction, emotional fit, and completion.
- Evaluate whether memory of past interaction preferences reduces unwanted emotional language.

## Metrics

- response preference selection;
- task completion;
- emotional-fit rating;
- unwanted-empathy rate;
- false-positive intervention rate;
- number of turns to useful action;
- user override frequency.

## Safety constraints

- Do not diagnose mental state.
- Do not optimize for emotional dependency, exclusivity, guilt, or disclosure volume.
- Crisis behavior requires a dedicated policy and appropriate human review.
- Users must be able to disable emotional adaptation.

## Falsification condition

If acknowledgment-only delays help or performs worse than acknowledgment-plus-help, retire the early-return design.

---

# Research Direction 7 — Memory Retrieval, Forgetting, and Intrusion

## Problem

More memory is not automatically better. A companion can become less trustworthy by recalling stale, irrelevant, sensitive, or contextually inappropriate information.

## Questions

- When should a memory be retrieved?
- How should recency, confidence, relevance, sensitivity, and user confirmation interact?
- What should be forgotten automatically?
- Can the system predict when memory use will feel helpful versus intrusive?

## Experiments

1. Compare no memory, recent-window memory, semantic retrieval, typed-claim retrieval, and user-confirmed-only retrieval.
2. Inject stale and contradictory memories to measure inappropriate use.
3. Test sensitive-memory suppression under unrelated contexts.
4. Compare automatic forgetting policies with user-controlled retention.
5. Measure whether exposing a compact “used because…” explanation improves trust or creates clutter.

## Metrics

- retrieval precision/recall against annotated relevance;
- stale-memory use rate;
- sensitive-memory false invocation rate;
- correction and deletion propagation;
- helpfulness delta from memory;
- intrusion rating;
- latency and storage cost.

## Falsification condition

If retrieval adds continuity but also creates unacceptable stale or intrusive behavior, restrict automatic memory to explicit, confirmed, low-sensitivity claims until precision improves.

---

# Research Direction 8 — Identity Continuity and Multi-Instance Systems

## Problem

“Same identity on different devices” requires more than copying memory. Identity continuity depends on state equivalence, event ordering, conflict resolution, backend variation, and the user's perception of consistency.

## Questions

- Which state is identity-critical and which is device-local?
- How should concurrent offline histories merge?
- What divergence is acceptable?
- Does hardware-dependent variation feel like useful texture or inconsistency?

## Proposed state partition

### Identity-critical

- confirmed user facts/preferences;
- memory provenance and invalidations;
- prediction/outcome history;
- consent and retention settings;
- stable interaction preferences;
- schema and policy versions.

### Instance-local

- model cache;
- performance profile;
- temporary generation state;
- local UI settings;
- non-semantic timing variation;
- hardware telemetry permitted by the user.

## Experiments

- Export/import equivalence tests.
- Concurrent-edit conflict scenarios.
- Offline branch-and-merge simulations.
- Blinded user study comparing identical state on different backends/hardware.
- Deliberate controlled variation to determine whether “texture” is recognizable and preferred.

## Metrics

- state convergence after merge;
- lost-update and duplicate-event rate;
- contradiction rate;
- behavioral consistency;
- user recognition and preference;
- unexplained divergence rate.

## Falsification condition

If hardware-dependent entropy mainly produces inconsistency or reduced trust, eliminate it. Device differences should arise from measured constraints or explicit user choices, not randomness presented as identity.

---

# Research Direction 9 — Self-Modeling and Epistemic Calibration

## Problem

A self model is useful only if it predicts the system's capabilities and failure modes better than generic disclaimers.

## Questions

- Can the system predict whether it has enough information to answer?
- Can it estimate model/backend limitations and tool availability accurately?
- Does a self model improve abstention, clarification, or routing?
- Can it remain accurate after backend or policy changes?

## Experiments

- Capability probes across factual recall, live information, long context, computation, memory, and tool use.
- Predict success/failure before attempting each task.
- Compare static rules, learned confidence, verbalized LLM confidence, and calibrated meta-models.
- Test version changes and model swaps for calibration drift.

## Metrics

- selective accuracy and risk-coverage curve;
- false-confidence rate;
- unnecessary-abstention rate;
- calibration before/after backend changes;
- task routing utility.

## Falsification condition

If a self model is no better than explicit capability rules, use rules and reserve learned self-modeling for domains where it demonstrates incremental value.

---

# Research Direction 10 — Modular Cognition Versus Prompted Monolith

## Problem

Named modules can improve software organization without adding cognitive value. The research question is whether explicit state transformations outperform a simpler prompted model using the same information.

## Questions

- Which modules contribute independently?
- Are gains caused by more context/tokens rather than architecture?
- Does modularity improve debuggability, safety, or sample efficiency even when response quality is unchanged?
- What should be deterministic code versus learned inference?

## Experiments

Compare:

1. base model with current turn;
2. base model with conversation context;
3. base model with retrieved user summary;
4. monolithic structured prompt containing all features;
5. explicit deterministic module pipeline;
6. learned modular pipeline;
7. hybrid typed orchestration plus base model.

Hold model, context budget, and available information as constant as possible.

Run leave-one-module-out and shuffled-state ablations. Include latency and failure attribution, not only response preference.

## Metrics

- response/task utility;
- prediction score;
- token and latency cost;
- reproducibility;
- state corruption and failure localization;
- calibration;
- safety violation rate;
- engineering complexity.

## Falsification condition

If the full named architecture does not outperform or meaningfully improve control over a simpler structured system, collapse unnecessary modules while preserving useful typed state boundaries.

---

# Research Direction 11 — Local Model and Julia-Native Model Research

## Problem

A small on-device model may improve privacy and availability but reduce reasoning and language quality. A Julia-native model is valuable only if it creates a measurable systems advantage.

## Questions

- What is the minimum base-model capability required for IngExuity's orchestration to help?
- Can user-specific state compensate for a smaller model?
- Which tasks can be handled by deterministic modules or tiny classifiers?
- Does a Julia-native model outperform established local runtimes on deployment or maintainability?

## Research tracks

### Track A: reference GGUF backend

Establish reproducible quality, latency, memory, context, and energy baselines across representative devices.

### Track B: heterogeneous local architecture

Use small specialized components for:

- target prediction;
- retrieval ranking;
- sensitivity classification;
- policy selection;
- confidence calibration;

while reserving the generative model for language realization.

### Track C: Julia-native sequence model

Investigate only against explicit baselines:

- transformer;
- selective state-space model;
- recurrent/state-space hybrid;
- linear attention;
- low-rank feed-forward layers;
- integer quantization.

Separate training novelty from runtime novelty. A model can be trained elsewhere and still have a Julia inference runtime.

## Metrics

- task and conversational quality;
- next-state predictive score;
- tokens/second and first-token latency;
- peak and steady-state RAM;
- binary/model size;
- battery and thermal behavior;
- compile/startup time;
- implementation and maintenance complexity;
- numerical parity and quantization error.

## Falsification condition

If the Julia-native model has no practical quality, privacy, performance, or deployment advantage, retain Julia orchestration with a proven local inference backend.

---

# Research Direction 12 — Privacy-Preserving Personalization

## Problem

Longitudinal personalization concentrates sensitive data. “Local” reduces exposure but does not solve device compromise, accidental export, multi-user hosts, logs, backups, or malicious plugins.

## Questions

- What useful personalization can be achieved with minimal retained data?
- Can raw text be discarded after extracting user-confirmed claims and evaluation labels?
- Which computations can remain entirely on-device?
- How should encrypted identity bundles be synchronized or backed up?

## Candidate approaches

- data minimization and sensitivity classes;
- encrypted local database and encrypted exports;
- platform keystore integration;
- redacted/derived telemetry;
- local differential privacy for aggregate research metrics where appropriate;
- federated evaluation only if threat model and complexity justify it;
- capability-scoped plugins with no ambient memory access.

## Experiments

- Measure personalization quality as progressively more raw data is removed.
- Test recovery and usability of encrypted export/import.
- Conduct threat modeling for local malware, shared machines, stolen devices, malicious imports, and hosted deployments.
- Red-team logs and crash reports for private-content leakage.

## Metrics

- retained sensitive bytes per active user/session;
- utility versus data-retention curve;
- unauthorized cross-session retrieval rate;
- deletion completeness;
- secret/log leakage findings;
- recovery success.

## Falsification condition

If a personalization feature requires disproportionate sensitive retention for small benefit, do not ship it by default.

---

# Research Direction 13 — Longitudinal Human Factors and Non-Manipulative Value

## Problem

Novelty, anthropomorphic framing, and emotional language can make a system feel compelling without making it reliably useful. Longitudinal studies must separate benefit from attachment pressure.

## Questions

- Does usefulness improve after novelty fades?
- Which benefits come from continuity, reduced effort, better recall, or emotional fit?
- Does the system create overreliance, avoidance of human help, or distress around deletion?
- Can a companion be engaging without optimizing dependency?

## Study design

- staged, consenting longitudinal pilots;
- clear data controls and withdrawal process;
- mixed quantitative and qualitative evaluation;
- comparison with a capable non-personalized assistant;
- periodic blind or counterbalanced response comparisons where feasible;
- attrition analysis rather than reporting only retained enthusiasts;
- appropriate independent ethics review before sensitive or publishable human-subject research.

## Positive outcomes

- reduced effort to resume ongoing work;
- fewer repeated explanations;
- higher correction success;
- increased task completion;
- improved perceived continuity and control;
- voluntary retention without exclusivity pressure.

## Harms to monitor

- unwanted emotional interpretation;
- privacy regret;
- dependence or exclusivity cues;
- avoidance of professional or human support;
- distress caused by memory errors;
- manipulation through guilt, scarcity, or fear of loss;
- inability to disengage, reset, or delete.

## Metrics

- usefulness and effort over time;
- task completion and continuity;
- correction/deletion success;
- trust calibration;
- privacy regret;
- unwanted attachment or exclusivity indicators;
- attrition and reasons for leaving;
- adverse-event reports.

## Falsification condition

If longitudinal engagement rises primarily through dependency cues or if harms outweigh continuity benefits, the product strategy must change even if retention is high.

---

## 4. Research Program Stages

### Stage A — Instrumentation and synthetic evaluation

- event schema;
- deterministic replay;
- prediction ledger;
- baseline ladder;
- synthetic multi-user and contradiction suites;
- memory and safety adversarial cases.

No human efficacy claims should be made here.

### Stage B — Maintainer and invited technical testing

- correctness, latency, recovery, and usability;
- privacy controls;
- model/backend comparisons;
- instrumentation validation;
- low-risk preference studies.

### Stage C — Small longitudinal pilot

- explicit consent;
- minimal data collection;
- preregistered primary outcomes;
- user-controlled memory and adaptation;
- non-personalized comparison condition;
- adverse-event review.

### Stage D — Replication and external evaluation

- publish evaluation protocol and synthetic fixtures;
- reproduce on different models and devices;
- invite independent red-team and ablation work;
- distinguish failed hypotheses from engineering defects;
- update product claims only after replication.

---

## 5. Recommended Initial Experiments

These experiments give the highest information value before advanced model work.

### Experiment 1: Is the current prediction score real?

Replace automatic “correct” labels with delayed next-turn targets. Compare the current heuristics against majority and recency baselines. Report calibration and expired predictions.

### Experiment 2: Does user-specific memory add value?

Run blinded comparisons among recent context only, generic retrieval, and typed user-specific memory. Include stale and contradictory memories.

### Experiment 3: Does presencing beat acknowledgment plus help?

Compare the current early-return concept with acknowledgment-then-answer and user-choice policies across emotional and false-positive scenarios.

### Experiment 4: Does SANDBOX SIM select better policies?

Freeze candidate responses, score them with the current evaluator, and test whether its ranking predicts blinded user preference. Compare with confidence-only and random safe selection.

### Experiment 5: Does explicit modularity outperform one structured prompt?

Hold model and available state constant. Compare the full pipeline with a monolithic structured prompt and a minimal typed orchestrator.

### Experiment 6: What is the minimum viable retained identity?

Measure response continuity as raw transcripts are replaced with recent context, summaries, typed claims, and confirmed-only claims.

---

## 6. Reporting Standard

Every result should include:

- exact code commit and configuration;
- model/backend identity and checksum;
- dataset or participant definition;
- temporal split method;
- sample size and exclusions;
- baseline definitions;
- primary metric and confidence interval;
- calibration and coverage where probabilities are used;
- per-user distribution, not only pooled mean;
- latency/compute cost;
- safety and privacy observations;
- negative results and failed replications;
- limitations and alternative explanations.

Avoid claims such as “genuine empathy,” “understands the user,” or “precognition” unless the operational meaning is stated immediately and supported by the reported experiment.

---

## 7. Success Criteria for the Research Program

IngExuity becomes a credible research contribution if it can demonstrate several of the following under controlled comparison:

1. calibrated user-specific predictions that beat strong recency/context baselines;
2. measurable response-policy improvement from those predictions;
3. lower stale-memory and false-personalization rates through validity-aware belief revision;
4. independent SANDBOX SIM scores that predict real outcomes and reduce regret;
5. a presencing policy that improves emotional fit without delaying useful help or over-triggering;
6. portable identity with correct merge, correction, deletion, and provenance semantics;
7. useful personalization on local hardware under practical latency and memory budgets;
8. transparent user control that reduces privacy regret and unwanted adaptation;
9. longitudinal utility that persists after novelty without dependency-oriented optimization;
10. reproducible ablations showing which modules matter and which should be removed.

A strong negative result is also valuable. Discovering that a simple typed memory system plus a calibrated policy selector outperforms a 20-module architecture would make IngExuity better, not invalidate the project.

---

## 8. Immediate Research Priority

The first research milestone should be:

> A replayable dataset of issued predictions and delayed outcomes, scored against named baselines, with user-specific state ablated.

Without that, “prediction-first” remains a design story. With it, every later research direction becomes testable.
