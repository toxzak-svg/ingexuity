# IngExuity Legacy Architecture Notes

**Status:** Legacy conceptual/prototype reference  
**Superseded by:** [`RUST_ARCHITECTURE.md`](RUST_ARCHITECTURE.md)  
**Production substrate:** Rust

This file formerly described the Julia prototype as the active IngExuity architecture. That is no longer accurate. The repository history preserves the complete earlier document; this page now records only the concepts that may still be evaluated during migration.

## Concepts retained for investigation

The following ideas remain research or product hypotheses, not proven production mechanisms:

- prediction-first interaction;
- explicit user modeling;
- temporal memory with validity and provenance;
- response-policy selection before generation;
- acknowledgment/presencing as a configurable policy;
- independent evaluation of candidate policies;
- identity portability across devices;
- local-first inference and storage.

## Concepts not automatically ported

The Rust implementation will not recreate the Julia module graph one file at a time. These names do not establish independent cognitive mechanisms by themselves:

- Comprehension;
- Self Model;
- User Model;
- Internal/Emotional;
- Curiosity;
- Research;
- Creative/Ingenuity;
- Decision;
- Precognition;
- Predictions;
- SANDBOX SIM;
- Action;
- Reaction Observance;
- Response;
- Voice;
- Output;
- Understanding;
- Intelligence.

A concept is implemented in Rust only when it has a typed contract, tests, a baseline, and a measurable contribution or necessary product responsibility.

## Legacy implementation facts

The retained Julia prototype uses or contains:

- HTTP.jl and JSON.jl for the current prototype server;
- LlamaCpp/GGUF integration;
- process-global conversation and emotional history state;
- in-memory memory storage;
- heuristic prediction and emotional-state logic;
- experimental or disabled custom-model paths.

These components are reference material only. New runtime, storage, API, evaluation, and inference-integration work belongs in the Rust workspace.

## Active architecture

See [`RUST_ARCHITECTURE.md`](RUST_ARCHITECTURE.md) for:

- active crates;
- session and state invariants;
- request flow;
- transactional failure behavior;
- prediction invariants;
- planned persistence, inference, evaluation, and safety boundaries;
- the Julia migration rule.

## Research program

See [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md) for hypotheses, experiments, metrics, baselines, ablations, and falsification conditions.
