# IngExuity Edge Pal Product Design

**Date:** 2026-07-13  
**Status:** Approved design direction  
**Figma:** [IngExuity — Edge Pal Product Architecture](https://www.figma.com/design/BAXFA8hSkXDEoR369w83PS)

## Product Promise

IngExuity is a local-first desktop pal that becomes more useful and more distinct through use. It combines an evolving personal relationship with permissioned desktop agency. The target is Claude-like reliability on the owner's daily work through a compound system—personalization, retrieval, tools, verification, and adaptive compute—not universal frontier-model parity from one small model.

The first supported hardware profile is a mainstream computer with 16 GB RAM and a CPU or integrated GPU. A paired desktop-and-Android experience is a planned expansion, not part of the first release.

## Relationship Model

Everyday Pal is the default relationship mode. Emotional Support and Productivity Partner are optional focus modes. Modes change priorities and behavior without creating separate identities or memory stores.

Adaptive Growth is independently toggleable in every mode. When enabled, nearly all learned behavior and identity expression may evolve. The user retains technical control through inspection, version history, named restore points, rollback, export, pause, and reset.

Behavioral state and remembered life history are versioned separately. A user can restore earlier behavior without erasing factual or episodic memory, or restore memory without silently changing the pal's current personality.

The adaptation boundary is intentionally loose. Strict harm prevention and the user's technical authority over the system remain invariant; warmth, agreeableness, opinions, humor, habits, relationship rituals, communication style, and other personality expression may change.

## Agency Model

Permission tiers are the default:

- Each tool category has an explicit capability grant and configurable limits.
- Consequential actions may require preview or confirmation based on category and risk.
- Broad Autonomy is an opt-in tier that may act first and report afterward within the user's configured boundaries.
- Protected resources, communication limits, spending limits, an emergency stop, and an immutable audit trail remain available in every tier.
- The assistant never hides or obstructs pause, inspection, export, rollback, or reset controls.

## System Architecture

### Desktop Pal

The desktop application owns chat, voice, notifications, tray presence, mode selection, growth controls, permission settings, activity review, and rollback. It presents one continuous identity across surfaces.

### Companion Kernel

The Companion Kernel is the system of record for identity and relationship behavior. It owns:

- the active relationship mode and mode-specific priorities;
- evolving behavioral and personality state;
- user preferences, routines, boundaries, important people, and projects;
- semantic, episodic, procedural, and prospective memory;
- behavioral versions, memory versions, and restore operations;
- unresolved conversational threads and current commitments.

The kernel exposes stable interfaces to the agent orchestrator. Model implementations do not own durable identity or memory.

### Agent Orchestrator

Nontrivial requests follow this loop:

1. Interpret the request and assess difficulty, risk, and latency needs.
2. Retrieve relevant personal context and external evidence.
3. Produce a structured plan when tools or multiple steps are required.
4. Check the plan against permissions and protected-resource rules.
5. Execute tools within the granted capability envelope.
6. Inspect outputs and deterministic checks.
7. Critique unsupported claims, incomplete work, and policy violations.
8. Repair failures or request user input when safe progress is impossible.
9. Respond with the result, material caveats, and an action receipt when applicable.

Simple conversation can bypass the full loop when the risk and difficulty classifiers both allow a direct response.

### Local Model Ladder

Production inference uses a replaceable runtime contract backed initially by `llama.cpp`:

- a fast conversational model for ordinary dialogue, routing, classification, and extraction;
- a stronger deliberate model for difficult reasoning, planning, coding, and sensitive responses;
- local embedding and reranking models for memory retrieval;
- specialist vision, speech, and verification models when installed;
- hardware-aware routing based on available memory, model residency, expected latency, and task difficulty.

The application ships hardware profiles rather than one universal model configuration. The 16 GB profile favors a small resident fast model and loads or invokes a stronger quantized model only when the expected quality gain justifies the latency.

### Tools and Verification

The first desktop tool families are files, browser, reminders, notifications, application launching, and drafted communications. Each tool publishes its risk class, reversibility, permission category, preview schema, and receipt schema.

Consequential output is checked by deterministic rules whenever possible. Examples include executing tests for code changes, validating file diffs, verifying that a reminder exists, checking message recipients, enforcing spending caps, and confirming that claimed tool results are present in returned evidence.

### Encrypted Personal Data

Personal data remains local by default in an encrypted vault. The vault stores memories, identity state, permissions, action receipts, and version history. Backups and device transfer use an encrypted identity bundle controlled by the user.

Cloud inference is not required by this design. A later opt-in provider may be added behind explicit per-category privacy rules without changing the local-first data model.

### Julia Research Track

The existing Julia-native model, prediction engine, emotional simulation, and edge-optimization work continue as an isolated research track. Experimental components cannot block the usable desktop product and cannot silently replace production components.

A research component is promoted only after it beats the production baseline on its intended workload. Promotion gates cover response or task quality, latency, peak memory, reliability, regression tests, and relationship-specific evaluations. A custom model does not pass merely because it runs or achieves acceptable perplexity.

## Data Flow

User intent enters through the Desktop Pal. The Companion Kernel supplies relevant identity, relationship, and memory context. The Agent Orchestrator chooses a direct response or a verified tool loop, and the Local Model Ladder supplies the models needed for each step. Permission checks precede consequential execution. Tool results, verifier evidence, and user reactions return to the orchestrator. Approved learning events update memory and behavioral state through versioned kernel transactions.

Research outputs flow only through benchmark and promotion gates; they never write directly into production state.

## Failure and Recovery

- Model failure falls back to a smaller resident model or a transparent degraded response.
- Tool failure returns a typed error and evidence; the orchestrator may retry only when the operation is safe and idempotent.
- Permission denial stops the affected action without degrading unrelated conversation.
- Memory conflicts are stored as unresolved assertions until corroborated or corrected.
- Behavioral regressions can be rolled back without deleting unrelated memories.
- Corrupt or incompatible state migrations restore from the last verified local snapshot.
- Broad Autonomy can be disabled immediately without waiting for the model runtime.

## Success Measures

The first release is judged on product outcomes rather than model novelty:

- accurate, non-creepy recall of user-provided information;
- successful restoration of behavioral and memory snapshots;
- high completion rate on supported desktop tasks;
- zero unauthorized consequential actions in the test suite;
- complete action receipts and permission decisions;
- interactive latency on the 16 GB reference machine for ordinary turns;
- clearly signaled deliberate-mode latency for difficult work;
- user-rated continuity, usefulness, and trust over repeated weeks of use.

## Delivery Sequence

1. Establish an executable baseline and hardware benchmark on the current repo.
2. Introduce stable runtime, kernel, storage, permission, and tool interfaces.
3. Ship the desktop shell with Everyday Pal, focus modes, and explicit growth controls.
4. Add versioned memory and behavioral rollback.
5. Add the compound local model ladder and retrieval pipeline.
6. Add permissioned desktop tools and deterministic verification.
7. Add Broad Autonomy as an advanced opt-in after permission and audit tests pass.
8. Continue Julia research behind reproducible promotion gates.
9. Prepare the encrypted identity bundle and protocol required for a later Android companion.

## Explicit Non-Goals for the First Release

- Universal parity with frontier cloud models on every benchmark.
- Android feature parity.
- Purchases or financial transactions.
- Unbounded background autonomy.
- Production dependence on the experimental Julia model.
- A second identity per focus mode.
