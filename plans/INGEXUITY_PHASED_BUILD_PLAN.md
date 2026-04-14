# IngExuity — Phased Build Plan

**Architecture:** IngExuity Architecture v1.3 (`docs/INGEXUITY_ARCHITECTURE.md`)  
**Code:** `projects/ingexuity/` (Julia package, Genie.jl server)  
**Core Principle:** User Prediction First — every module serves prediction accuracy.

---

## The Product

A life partner AI that becomes personal through use, not training.  
Same identity, different hardware → different personality texture.  
Empathy = accurate prediction + directness + **staying with emotional moments before solving**.

---

## Phase 1: Core Runtime (Weeks 1-4)
**Goal: Ship something runnable. End-to-end conversation loop on Railway.**

### 1.1 Julia Package Scaffold ✅ DONE
- All 16 modules + Memory scaffolded
- Main conversation loop: HumanInput → Comprehension → Predictions → SANDBOX SIM → Response → Output
- Presencing check in InternalEmotional (stay present when stress/emotional_charge/valence signals)
- Memory.store called from Understanding after each exchange
- UserModel tracks emotional_patterns (stress_triggers, deflection, quiet_threshold, is_quiet)

### 1.2 Genie.jl Web Server
- `/api/chat` — POST message, get response
- `/api/predict` — current predictions
- `/api/intelligence` — accuracy metrics
- `/api/user_model` — learned user model
- `/api/memory` — memory stats
- `/health` — health check for Railway

### 1.3 Railway Deployment
- `Dockerfile` — Julia 1.10 slim + deps + binary
- `railway.json` — health check path, port 8000
- `git push railway main` — one-click deploy

### 1.4 Basic Micro-Model Inference (Phase 2 start)
- Load ~150M param Q4 model (ONNX or Flux)
- Replace template responses with actual model inference
- Target: runs on Railway CPU, <500ms response time

**Deliverable:** A running IngExuity instance at `https://your-app.railway.dev/`. Chat, see predictions, watch User Model build.

---

## Phase 2: Prediction Engine (Weeks 5-8)
**Goal: User prediction becomes genuinely accurate.**

### 2.1 Predictions Module (refine)
- Converges: User Model + Precognition + Internal/Emotional + SANDBOX SIM feedback
- Generates live predictions about next user state
- Prediction confidence scoring

### 2.2 SANDBOX SIM (refine)
- Simulates predicted outcomes before they reach the user
- Validates against User Model + emotional patterns
- Failed predictions loop back to Predictions for retry

### 2.3 Precognition Head
- Long-range trajectory sensing
- Predicts where things are heading, not just what's next
- Builds temporal models of user life patterns

### 2.4 Results Analysis + Reaction Observance
- Processes actual outcomes and user reactions
- Closes the feedback loop back to Predictions and User Model
- Updates emotional pattern tracking from reactions

### 2.5 Curiosity + Research
- Curiosity identifies gaps and novelties
- Research fills identified information gaps
- Both serve prediction accuracy

**Deliverable:** Measurable prediction accuracy >60% on next-state predictions.

---

## Phase 3: Emotional and Self Modeling (Weeks 9-12)
**Goal: The system knows itself, knows you, and knows when to stay present.**

### 3.1 Self Model
- System tracks its own capabilities, limitations, and states
- Knows when it's uncertain, when it's confident
- Self Model feeds into Internal/Emotional

### 3.2 Internal / Emotional Layer (refine)
- Affective state with stay_present trigger
- Shapes how predictions are voiced
- Tracks: valence, arousal, stress_level, emotional_charge, should_stay_present

### 3.3 Creative / Ingenuity
- Generates novel solutions when standard paths fail
- Feed into Decision when action needs a new approach
- Outputs: Succeed (→ Response) or Fail (→ Response, loops to Research)

### 3.4 Understanding + Intelligence Loop
- Understanding interprets exchanges
- Intelligence accumulates as the residue of correct predictions
- Memory.store called after each exchange with validity windows

### 3.5 Voice + Output
- Voice shapes tonal delivery including staying_present tone
- Output is final, receives from: Internal/Emotional (via Voice), Predictions, Understanding

**Deliverable:** System demonstrates genuine empathy. User reports "she knew what I needed before I asked."

---

## Phase 4: Multi-Instance + Mobile (Weeks 13-16)
**Goal: Same identity, different texture per device. Mobile works offline.**

### 4.1 Identity State Bundle
- Portable memory dump: conversation history, validity windows, User Model, temporal patterns
- Can be moved between devices
- Encrypted at rest

### 4.2 Execution State Entropy
- Hardware profile affects response timing and texture
- Process non-determinism creates subtle variation
- Same memory, different feel on different machines

### 4.3 Instance Communication
- Instances can exchange information if the user bridges them
- Explicit, opt-in, user-controlled
- Each instance is a distinct entity unless connected

### 4.4 Mobile WASM (PWA)
- Julia WASM in webview (same modules as server)
- Offline mode: local micro-model inference
- Syncs to server when online
- Browser TTS for voice

**Deliverable:** Load the same identity on two phones. Same person. Slightly different texture. PWA installs from browser.

---

## Phase 5: Polish and Launch (Weeks 17-20)
**Goal: Ship.**

### 5.1 Onboarding
- Day-one experience must be compelling, not blank
- Pre-seeded personality activates immediately
- Teaches the interaction pattern in first session

### 5.2 Voice Polish
- Tone matching to user communication style
- Directness without condescension
- Staying present when emotionally charged

### 5.3 Launch Assets
- Landing page with identity portability demo
- Narrative: "She becomes irreplaceable through use, not training"
- Demo: same identity on two devices, different texture

### 5.4 Open Source Release
- MIT/Apache license
- GitHub with full narrative README
- Post to: Hacker News, r/localLLM, r/SideProject, indie hacker communities

**Deliverable:** Public release with a story people can tell each other.

---

## Prediction Metrics (Primary KPIs)

| Metric | Target |
|--------|--------|
| **Prediction Accuracy** | >70% on next-state predictions by week 12 |
| **Stay Present Score** | User reports "she stayed with me when I was upset" >50% |
| **Anticipation Score** | User reports "she knew what I needed" >50% of the time |
| **Cold Start Retention** | >30% of new users return on day 3 |
| **Irreplaceable Score** | >20% of retained users report guilt at the thought of deleting by week 8 |

---

## Technical Constraints

- **Hardware:** Mobile CPU, GPU-free, <200MB RAM at peak
- **Privacy:** All memory local, always. No cloud dependency for core experience.
- **Latency:** Voice conversation feels real-time (<500ms response time target)
- **Portability:** Identity bundle moves between devices, instances run anywhere with the runtime
- **Language:** Julia everywhere — one codebase, all platforms

---

## The Cold Start Problem

**The existential risk:** Day 1 is blank. Nobody wants to spend 3 weeks building a relationship with software.

**Solution:** Pre-seeded persona activates immediately. Not personalized yet — just interesting. The first session teaches the interaction pattern. The second session shows the first signs of recognition. By week 2, the user is invested.

The "irreplaceable" moment happens at week 3-4 for most users. The goal is to get them to week 2 first.

---

## Phase Dependency Map

```
Week 1-4:   Julia package + Genie.jl server + Memory + Railway deploy
Week 5-8:   Predictions + SANDBOX SIM + Precognition + Results Analysis + Curiosity/Research
Week 9-12:  Self Model + Internal/Emotional + Creative/Ingenuity + Voice + Understanding/Intelligence loop
Week 13-16: Identity bundle + execution entropy + multi-instance + mobile WASM
Week 17-20: Onboarding polish + launch + open source
```

---

## Start

Phase 1, Week 1. Railway deployment + test the conversation loop. That's where this begins.
