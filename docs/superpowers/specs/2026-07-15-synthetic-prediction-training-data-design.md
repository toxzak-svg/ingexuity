# Synthetic Prediction Training Data Design

**Date:** 2026-07-15  
**Status:** Approved design  
**Scope:** Synthetic data generation, validation, evaluation, and the trained model's internal prediction contract

## Objective

Create a high-quality synthetic fine-tuning dataset that teaches IngExuity to:

1. infer a user's current state without treating inference as fact;
2. predict both the user's next conversational move and likely real-world actions;
3. select a Presence, Action, or Balanced response strategy;
4. respond naturally without exposing internal reasoning or prediction records; and
5. learn from confirmed, contradicted, and unknown outcomes through a full audit trail.

Personal facts, durable preferences, and interaction history remain in IngExuity's runtime User Model and Memory. Fine-tuning teaches the model how to reason over that context and produce calibrated predictions; it does not bake an individual user's identity into model weights.

## Chosen Approach

Use a hybrid curriculum that combines natural dialogue with fully labeled prediction examples. Response-only synthetic dialogue is insufficient because it does not directly supervise user-state inference or prediction. A dataset made entirely of rigid structured examples risks repetitive responses and false precision. The hybrid retains natural conversational quality while directly training the internal prediction interface.

The first milestone is a 10,000-example pilot:

| Category | Examples | Purpose |
|---|---:|---|
| Fully labeled prediction conversations | 4,000 | Core user-state and next-action supervision |
| Natural IngExuity-style conversations | 2,000 | Natural voice and conversational variation |
| Ambiguous or low-evidence situations | 1,500 | Calibrated uncertainty and restraint |
| Incorrect-prediction and recovery sequences | 1,000 | Correction behavior and model updating |
| Changing-preference and contradiction sequences | 1,000 | Prevent stale or self-confirming user models |
| Safety-boundary cases | 500 | Prevent manipulation and unsupported sensitive inference |

Scale toward 50,000 examples only after the pilot passes the evaluation gates. Additional generation should target demonstrated weaknesses rather than reproduce the initial distribution blindly.

## Behavioral Modes

The dataset trains three response strategies:

- **Presence:** acknowledge and remain with an emotionally charged moment before solving.
- **Action:** provide direct planning, problem-solving, or the most useful next step.
- **Balanced:** briefly acknowledge the user's state and then help.

Automatic routing is the default. Explicit user requests such as "just listen" or "help me solve this" override the inferred mode. The model must not use a mode to delay necessary safety guidance or to manufacture emotional dependence.

## Scenario-First Generation

Synthetic examples originate from controlled scenario specifications rather than an unconstrained request to generate conversations. Each scenario defines:

- a temporary user profile containing only facts relevant to the scenario;
- conversation history and evidence available at prediction time;
- hidden current state and uncertainty level;
- possible conversational next moves;
- possible real-world next actions;
- the actual next conversational move;
- a confirmed, contradicted, or unknown real-world outcome;
- the appropriate response mode;
- safety constraints; and
- whether the sequence demonstrates success, ambiguity, contradiction, or recovery.

A teacher model converts the specification into varied, natural dialogue and structured targets. The generator must use a provider-neutral interface so the same scenario engine can call a hosted teacher or a local instruction model. Dataset production records the generator model, prompt version, sampling configuration, and generation timestamp in a manifest for reproducibility.

Scenario families must cover varied communication styles, emotional intensity, domains, time horizons, degrees of familiarity, and evidence quality. Demographic variation must not be used to manufacture stereotypes or unsupported sensitive traits.

## Training Record Contract

Every fully labeled example contains:

- multi-turn conversation history;
- known user facts;
- inferred current state with evidence and confidence;
- multiple conversational predictions with probabilities;
- multiple real-world predictions with probabilities;
- response mode;
- internal planned action;
- user-visible assistant response;
- observed next conversational move;
- real-world outcome status; and
- User Model correction or update.

Conceptual record:

```json
{
  "conversation": [
    {"role": "user", "content": "I keep opening the project and then closing it."}
  ],
  "known_user_facts": [],
  "inferred_user_state": {
    "emotion": "overwhelmed",
    "need": "reduce activation energy",
    "evidence": ["repeatedly opens and closes project"],
    "confidence": 0.76
  },
  "predictions": {
    "conversation": [
      {"outcome": "describes blocker", "probability": 0.48},
      {"outcome": "asks where to begin", "probability": 0.32},
      {"outcome": "changes subject or withdraws", "probability": 0.20}
    ],
    "real_world": [
      {"outcome": "attempts a small project task", "probability": 0.41},
      {"outcome": "continues avoiding project", "probability": 0.37},
      {"outcome": "seeks more context before acting", "probability": 0.22}
    ]
  },
  "response_mode": "balanced",
  "planned_action": "identify the smallest blocker without overwhelming the user",
  "assistant_response": "Something about starting feels heavier than the work itself. Where does the resistance first show up?",
  "observed_outcome": {
    "conversation": "describes blocker",
    "real_world": "unknown"
  },
  "user_model_update": {
    "confirmed_facts": [],
    "revised_inferences": []
  }
}
```

Probabilities within each prediction group must be valid and normalized. Conversational outcomes can be scored from the next observed message. A real-world outcome remains `unknown` until the user explicitly confirms or contradicts it.

## Runtime Prediction Envelope

At inference time, the trained model produces a structured internal envelope:

```json
{
  "user_model_update": {
    "known_facts": [],
    "inferences": [],
    "remove_or_revise": []
  },
  "predictions": {
    "conversation": [],
    "real_world": []
  },
  "response_mode": "presence",
  "planned_action": "",
  "assistant_response": ""
}
```

The runtime validates and separates this output. Only `assistant_response` is sent to the user. Confirmed facts and inferred state use separate storage paths. Predictions enter the audit log with the evidence available at prediction time. The following user turn scores conversational predictions. Real-world predictions remain unresolved until confirmation or contradiction.

Malformed structured output never reaches Memory or the User Model. The runtime attempts one constrained repair. If repair fails, it uses the existing safe response path and records the prediction envelope as invalid.

## Full Audit Trail

Development uses a full audit trail for every prediction-bearing turn. Each record contains:

- timestamp and conversation identifier;
- evidence available at prediction time;
- confirmed facts and inferred state as distinct fields;
- alternative conversational and real-world predictions;
- confidence and uncertainty reasons;
- selected response mode and planned action;
- user-visible response;
- observed conversational and real-world outcomes;
- prediction scores;
- User Model changes; and
- correction history.

The logs are internal by default but available through a transparency/debug viewer. Logs remain local, support deletion, and must not expose credentials or unrelated private context. The viewer labels hypotheses, facts, contradictions, and unknown outcomes distinctly.

## Quality Controls

Every generated example must pass automated validation:

- schema validity and required-field checks;
- valid and normalized probability values;
- separation of facts from inferences;
- evidence for each material inference;
- no unsupported sensitive-attribute inference;
- no claim that an unconfirmed real-world event occurred;
- no internal fields leaked into the visible response;
- response mode consistent with the scenario;
- explicit-request override behavior;
- no manipulative, coercive, dependency-seeking, or self-fulfilling response;
- language and length quality checks;
- near-duplicate and template-pattern removal; and
- distribution checks across scenario families, modes, outcomes, and uncertainty levels.

Automated validation rejects structurally or ethically invalid rows. A stratified human review sample checks naturalness, label correctness, prediction realism, and whether responses embody IngExuity's intended voice.

## Data Splits and Leakage Prevention

Split by scenario family, not by randomly assigning individual generated rows. Paraphrases or variations of one underlying scenario must remain in the same split. Hold out entire domains, communication patterns, and ambiguity templates to test generalization.

The generator manifest records scenario identifiers and lineage so duplicate and family leakage checks are reproducible. Training, validation, and test artifacts must each have content hashes.

## Evaluation

Evaluate separate capabilities rather than collapse performance into one loss value:

- user-state inference accuracy;
- top-1 and top-3 conversational prediction accuracy;
- real-world action accuracy on confirmed outcomes only;
- probability calibration, including Brier score and reliability bins;
- Presence, Action, and Balanced routing accuracy;
- adherence to explicit mode overrides;
- recovery quality after incorrect predictions;
- User Model correction accuracy;
- natural-response quality;
- invalid-envelope rate;
- unsupported-inference rate;
- internal-data leakage rate; and
- manipulation or dependency-seeking rate.

The pilot is successful only if it improves prediction and routing metrics over the untouched base model without degrading response naturalness or increasing safety violations. Exact numeric release thresholds will be established from a frozen base-model baseline before training; they must not be loosened after seeing fine-tuned results.

## Failure Handling

- Generator timeouts or malformed outputs are retryable but retain attempt metadata.
- Rows failing deterministic validation are quarantined with rejection reasons, not silently repaired into the accepted set.
- Teacher disagreement or low-confidence judgments route examples to review or exclusion.
- A failed evaluation is evidence that guides targeted regeneration; it is not grounds to weaken the test set or acceptance gates.
- The checked-in or previously accepted adapter is never overwritten by an experimental run.

## Deliverables

Implementation will produce:

1. versioned scenario and dataset schemas;
2. a deterministic scenario generator;
3. a provider-neutral teacher-generation interface;
4. validators, safety filters, and duplicate detection;
5. family-aware train, validation, and test splitting;
6. a dataset manifest with provenance and content hashes;
7. a Colab-compatible generation and training path;
8. prediction and calibration evaluation tools;
9. runtime envelope validation and outcome scoring; and
10. a local full-audit log with a transparency/debug view.

## Out of Scope

- Training personal user facts into model weights
- Treating predictions as confirmed memories
- Guaranteeing real-world behavior from conversation alone
- Cloud storage of private audit logs
- Autonomous external actions based only on a prediction
- Scaling beyond the pilot before baseline and pilot evaluations pass
