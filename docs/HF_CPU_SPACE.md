# Hugging Face CPU Basic Space

## Purpose and boundary

The CPU Space is a free, sleeping, single-generation evaluation profile. It is not a production latency service-level objective. It serves `bartowski/Llama-3.2-1B-Instruct-GGUF` using `Llama-3.2-1B-Instruct-Q4_K_M.gguf`; it does not serve the Gemma 4 12B LoRA candidate and must not be described as doing so.

The released defaults are:

- 4,096-token context
- 2 inference threads
- 1 active generation slot
- 1,024 default output tokens
- 2,048 maximum output tokens
- 160 tokens reserved for chat-template overhead

The adapter preserves the newest turns and reports `context_trimmed: true` whenever older turns are omitted. A response stopped by the configured output limit is reported as `finish_reason: "length"` and may be continued with `/api/chat/continue`. A normal end-of-sequence response is reported as `stop` and cannot be continued automatically.

## Runtime inspection

`GET /api/runtime` exposes the model and revision, context and output limits, thread count, process state, sanitized child logs, and the latest numeric request timing. It never includes conversation contents.

`GET /health` returns 200 only after `llama-server` has loaded the model and completed a one-token warm-up.

## Publish

Install the release tooling in a clean environment:

```bash
python -m pip install "huggingface_hub==0.34.4" "requests==2.32.4" pytest
```

Set a write-capable token without placing it in command history:

```bash
export HF_TOKEN="..."
```

Run the complete local suite, then publish the exact reviewed Git revision:

```bash
python -m pytest spaces/hf_cpu/tests scripts/test_benchmark_hf_cpu.py \
  scripts/test_publish_hf_cpu_space.py \
  -q --basetemp=.pytest-tmp-hf-cpu-final -p no:cacheprovider
python scripts/publish_hf_cpu_space.py \
  --repo-id toxzak/ingexuity \
  --revision "$(git rev-parse HEAD)"
```

The publisher atomically replaces the remote Space source with only the reviewed Docker Space manifest, removing stale GPU-sidecar files. It excludes tests, caches, checkpoints, adapters, GGUF files, safetensors, and binaries. It does not alter Space hardware, secrets, visibility, or sleep settings.

Hugging Face Docker build secrets must include `HF_TOKEN` when the model download requires authenticated license access. The Dockerfile consumes it through a BuildKit secret mount, never through an image layer or build argument.

## Benchmark

Run the benchmark only against the deployed Space:

```bash
python scripts/benchmark_hf_cpu.py \
  --base-url https://toxzak-ingexuity.hf.space \
  --out artifacts/hf_cpu_benchmark.json
```

When `HF_TOKEN` is present, the benchmark sends it as a bearer header so the same command works against a private Space. Public Spaces require no token. The credential is redacted from captured errors and is never written to the report.

The script performs one excluded warm-up, then three samples each for a 256-token short case and a 1,024-token long case. It records time to first token, wall time, completion tokens, decode tokens per second, finish reason, trimming, and continuation success. It stores no authorization token or prompt response text.

Do not claim a speed multiplier from a local machine. The `2.5x` objective is accepted only when the deployed CPU Basic artifact demonstrates it. If it does not, describe this profile as a functional CPU fallback.

## Tuning rules

Change one dimension per deployed benchmark:

| Variant | Threads | Context | Parallel | Rule |
| --- | ---: | ---: | ---: | --- |
| A | 2 | 4096 | 1 | Baseline |
| B | 1 | 4096 | 1 | Keep only for at least 10% decode improvement with no TTFT regression |
| C | 2 | 8192 | 1 | Keep only when required and within 10% of A decode rate |
| D | 2 | 4096 | 2 | Reject on queuing, OOM, or per-request regression |

A larger output limit increases total waiting time. A larger context can reduce prompt and decode performance. Restore the last verified profile after every rejected experiment.

## Rollback

1. Open the Space repository history and identify the last known-good Hub commit.
2. Check out or download that revision.
3. Upload its reviewed source files as a new rollback commit, or use the Hub UI to revert the failed commit.
4. Wait until the Space stage is `RUNNING`.
5. Verify `/health`, `/api/runtime`, a streamed 256-token request, and a length-stopped continuation.

The Space sleeps on free hardware. A cold request may initially receive a platform wake-up page or a loading state. Retry only after the Space reports `RUNNING`; do not interpret cold-start time as post-warm decode performance.

## License notice

The Space UI and repository display “Built with Llama” and include the required Llama 3.2 attribution notice. Preserve `NOTICE` and the README attribution in every release.
