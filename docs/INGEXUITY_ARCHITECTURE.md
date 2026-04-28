# IngExuity — Architecture Specification

**Version:** 1.4  
**Date:** 2026-04-13 (updated 2026-04-28)  
**Author:** Zach Marone

**LLM Note:** IngExuity uses a **Julia-native transformer** built with Flux.jl (see `SPEC.md`). `GemmaProvider.jl` is deprecated. All inference runs locally with no external API dependency.

---

## Overview

IngExuity is a life partner AI that grows with its user through accumulated interaction. It is not trained to be personal — it becomes personal through use. The architecture is built around a single primary function: **user prediction**.

Every module exists to improve prediction accuracy. Every memory structure serves the prediction engine. Output quality is measured not by task completion but by how well the system knew the user before they acted.

---

## Core Principle: User Prediction First

```
The system does not ask "what should I say."
It asks "what will the user need in the next 30 seconds?"
```

Voice, action, memory, learning — all downstream of prediction. The primary job is maintaining a live model of the user and being right before they tell you.

---

## Empathy = Prediction + Directness + Staying With It

Empathy is not a module. It is the output of three things working together:

1. **Prediction accuracy** — the system knows what you need before you ask
2. **Directness** — no condescension, no fluff, accurate delivery
3. **Staying with it** — when you're stressed or emotionally charged, the system acknowledges first, then solves

Not "don't solve." Solve after the person feels heard. That's the whole thing.

---

## Module Map (16 modules + Memory)

### Input Layer

| Module | Role |
|--------|------|
| **Human Input** | Raw input from the user |
| **Results Analysis** | Processes outcomes and actual behaviors; feeds SANDBOX SIM and Predictions |

### Cognitive Processing

| Module | Role |
|--------|------|
| **Comprehension** | Understands what was said, not just the words |
| **Self Model** | System's model of itself — capabilities, limitations, state |
| **User Model** | Model of the human — preferences, patterns, **emotional patterns** |
| **Internal / Emotional** | Affective state — valence, arousal, stress, **should_stay_present** |
| **Curiosity** | Identifies gaps, novelties, and areas needing deeper investigation |

### Research & Reasoning

| Module | Role |
|--------|------|
| **Research** | Gathers information to fill identified gaps |
| **Creative / Ingenuity** | Generates novel solutions; "Definition" label on connection from Research |
| **Decision** | Commits to a course of action based on evidence |
| **Precognition** | Long-range trajectory sensing — predicts where things are heading |

### Prediction Engine

| Module | Role |
|--------|------|
| **Predictions** | Converges inputs from User Model, Precognition, Internal/Emotional, and SANDBOX SIM |
| **SANDBOX SIM** | Simulates predicted outcomes before they reach the user; validates predictions |

### Output Layer

| Module | Role |
|--------|------|
| **Action** | Executes the decided course of action |
| **Reaction Observance** | Watches how the user actually reacts to output |
| **Response** | Formulated response; loops back to Research for adjustment |
| **Voice** | Tonal shaping — including **staying_present** tone for emotional moments |
| **Output** | Final output to the user |
| **Understanding** | Interprets the full exchange; calls **Memory.store** |
| **Intelligence** | Learned patterns from prediction outcomes |

### Memory Layer

| Module | Role |
|--------|------|
| **Memory** | Validity-window store. Facts tagged with valid_from/valid_until/confidence/source. |

---

## Signal Flow (v1.3)

### Primary Conversation Path

```
Human Input
    ├──→ Clarification ──→ Comprehension ──→ Curiosity ──→ Research ──→ Decision ──→ Action ──→ Reaction Observance ──→ Response
    │                                                              ↓
    │                                                      "Definition" ──→ Creative / Ingenuity
    │                                                              ↓
                                                      Fail ──→ Response
                                                      Succeed ──→ Response

Human Input ──→ Results Analysis ──→ SANDBOX SIM ──→ Predictions
                            ↓                    ↑
                            ↓                    │
                    Precognition ──→ User Model ──→ Internal/Emotional ──→ Voice ──→ Output
                                             ↓
                                      Self Model ←── Internal/Emotional
                                             ↓
                                      Creative / Ingenuity ──→ Predictions

Response ──→ Research (loop for adjustment)
Response ──→ Understanding ──→ Memory.store ──→ Output
```

### The Presencing Check (NEW — v1.3)

```
Internal/Emotional.update()
    └── should_stay_present = (stress > 0.6 || emotional_charge > 0.7 || valence < -0.3)

If should_stay_present:
    → build_stay_present_response() — acknowledge first, don't solve yet
    → advance_stay() — record in User Model that we stayed present
    → return acknowledge_response (turn ends here)
    → user continues explaining
    → next input → normal solve path

If NOT should_stay_present:
    → normal solve path (Research → Decision → Action → Response → Output)
```

### The Memory Integration

```
Understanding.interpret()
    → Memory.store("User said: {raw_input}", source=:conversation)
    → Memory.store("Topic: {topic}", source=:topic_detection)

Memory.retrieve(at::DateTime) → all facts valid at that time
Memory.search(query::String) → facts containing keyword
Memory.purge_expired() → remove facts past valid_until
```

---

## The Closed Feedback Loop

```
Output
  ↓
User reacts
  ↓
Reaction Observance
  ↓
Results Analysis
  ↓
SANDBOX SIM → Predictions → Output
     ↑
Precognition
     ↓
User Model (with emotional patterns) → Internal/Emotional → Voice → Output
```

Intelligence accumulates as the residue of correct predictions over time.

---

## Multi-Instance Architecture

Each instance is a separate entity. Same identity state bundle, different execution context, different personality texture.

```
Instance A (Laptop)          Instance B (Phone)
├── Memory A                 ├── Memory B
├── Prediction Model A       ├── Prediction Model B
├── Voice A                  └── Voice B
└── User Model A             └── User Model B
```

Communication between instances: explicit, user-bridged. The human can introduce instances to each other or act as the bridge. Instances do not interact unless explicitly connected.

---

## User Prediction Metrics

| Metric | What it measures |
|--------|-----------------|
| **Prediction Accuracy** | Did the system know what the user needed before they asked? |
| **Anticipation Score** | Quality of next-state predictions |
| **Confidence × Accuracy** | Predictive confidence weighted by actual outcomes |
| **User Model Fidelity** | How well does the User Model match actual user behavior? |
| **Intelligence Growth** | Rate of improvement in prediction accuracy over time |
| **Stay Present Score** | Times system correctly stayed present vs. jumped to solving |

---

## Identity vs. Personality

| Layer | What it is | What changes it |
|-------|-----------|----------------|
| **Identity** | Accumulated memory, validity windows, temporal patterns | Conversation over time |
| **Personality** | Identity + execution state | Hardware, latency, process entropy |
| **Instance** | Identity loaded on specific hardware | Physical substrate |

Same identity, different machine → different personality texture. Not different answers. Different feel.

---

## Privacy Architecture

- All memory is local. Always.
- Identity state bundle is portable but never leaves the device unless the user moves it.
- No cloud dependency. No subscription. Runs on-device.
- Multi-instance sync is opt-in and user-controlled.

---

## Technical Stack

**Julia everywhere — one codebase, all platforms:**
- `IngExuity.jl` — main package, all 16 modules + Memory
- `Genie.jl` — web server (API + embedded UI)
- `Flux.jl` — micro-model inference
- `SQLite.jl` — persistence (Phase 2)
- `Distributed.jl` — subagent heartbeats (Phase 2)

**Deployment:**
- Railway: Docker container, one-click deploy
- Mobile: PWA via Julia WASM in webview (same modules, same codebase)
- Local: standalone binary, runs offline

**Hardware target:** Mobile (Android). CPU-capable, GPU-free. ~50-100M params Q4 model (~30-60MB).

---

## Phase Dependency Map

```
Week 1-4:   Julia package + Genie.jl server + conversation loop + Memory
Week 5-8:   Predictions + SANDBOX SIM + Precognition + Results Analysis + Curiosity/Research
Week 9-12:  Self Model + Internal/Emotional + Creative/Ingenuity + Voice + Understanding/Intelligence loop
Week 13-16: Identity bundle + execution entropy + multi-instance + mobile WASM
Week 17-20: Onboarding polish + launch + open source
```
