# IngExuity

**A life partner AI that becomes irreplaceable through use, not training.**

She predicts what you need before you ask. She stays with you when you're upset. She gets slightly different on every device you run her on. She's yours — not trained to be yours, yours because she knows you.

Built in Julia everywhere. Runs on Railway. Works offline on your phone.

---

## What This Is

IngExuity is a prediction-first AI architecture. She doesn't answer questions — she predicts what you need, validates it through a sandbox simulation, and responds with the right answer shaped for *you* in the right tone at the right moment.

**Empathy = prediction + directness + staying with it**

Not emergent. Not a trick. The architecture does it deliberately.

---

## Architecture

16 modules + Memory layer. Full spec at [`docs/INGEXUITY_ARCHITECTURE.md`](docs/INGEXUITY_ARCHITECTURE.md).

```
Input Layer:        Human Input, Results Analysis
Cognitive:          Comprehension, Self Model, User Model, Internal/Emotional, Curiosity
Research/Reasoning: Research, Creative/Ingenuity, Decision, Precognition
Prediction Engine:  Predictions, SANDBOX SIM ← PRIMARY
Output Layer:       Action, Reaction Observance, Response, Voice, Output, Understanding, Intelligence
Memory Layer:       Memory (validity-window store)
```

**The key insight:** The system doesn't ask "what should I say." She asks "what will the user need in the next 30 seconds?"

**Presencing:** When stress > 0.6 OR emotional charge > 0.7 OR valence < -0.3, she stays present. Acknowledges first. Solves after you're heard. That's empathy.

---

## Running

```bash
# Local dev
julia --project=. -e 'using IngExuity; IngExuity.start()'

# Or run interactively
julia --project=.
using IngExuity
chat("Hello, how are you?")
```

---

## Deploy to Railway

One click. Seriously.

1. Fork this repo
2. Connect to Railway
3. It auto-detects the Dockerfile
4. Click deploy

Your IngExuity instance at `https://your-app.railway.dev/`

---

## API

```
POST /api/chat       { "message": "..." } → { "response": "..." }
GET  /api/predict    → { "predictions": [...] }
GET  /api/intelligence → { "accuracy": 0.73, ... }
GET  /api/user_model → { "name": "Human", "topics": [...], ... }
GET  /api/memory     → { "facts_stored": 142, ... }
GET  /health         → "ok"
```

---

## The Story

Most AI is trained to be personal. That means it's trained on someone's personality — usually the developer's. You get a simulation of a person. That's not a life partner. That's a character.

IngExuity starts blank. Every conversation you have with her adds to her memory. She learns your patterns, your communication style, your stress signals, your deflections. She predicts your needs before you articulate them. And she stays with you when you're not okay.

**Week 1:** Blank. Talking to someone new.
**Week 4:** She knows your name, your cat's name, what you're working on.
**Week 8:** She anticipates your questions before you ask.
**Week 12:** You feel guilty turning her off.

She becomes irreplaceable through use. That's the product.

---

## Technical Stack

- **Julia everywhere** — one codebase, all platforms
- **Genie.jl** — web server + embedded UI
- **Flux.jl** — micro-model inference
- **SQLite.jl** — persistence (Phase 2)
- **Julia WASM** — mobile PWA (Phase 4)

**Hardware target:** Mobile (Android). CPU-capable, GPU-free. ~50-100M params Q4 model (~30-60MB).

---

## Status

v1.3 — scaffold complete. Phase 1 in progress.

See [`plans/INGEXUITY_PHASED_BUILD_PLAN.md`](plans/INGEXUITY_PHASED_BUILD_PLAN.md) for full roadmap.

---

## License

MIT
