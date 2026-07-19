#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterator

import requests

CASES = [
    {"id": "short", "messages": [{"role": "user", "content": "Give three concise planning tips."}], "max_new_tokens": 256},
    {"id": "long", "messages": [{"role": "user", "content": "Write exactly 180 numbered launch-checklist items. Each item must be one complete sentence of 8 to 14 words. Do not summarize, skip numbers, or stop before item 180."}], "max_new_tokens": 1024},
]


@dataclass
class Sample:
    case_id: str
    success: bool
    request_id: str | None = None
    first_token_seconds: float | None = None
    wall_seconds: float | None = None
    completion_tokens: int | None = None
    decode_tokens_per_second: float | None = None
    finish_reason: str | None = None
    context_trimmed: bool | None = None
    continuation_ok: bool | None = None
    error: str | None = None


def build_session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    resolved_token = os.environ.get("HF_TOKEN", "") if token is None else token
    if resolved_token:
        session.headers.update({"Authorization": f"Bearer {resolved_token}"})
    return session


def scrub_session_secrets(text: str, session: requests.Session) -> str:
    authorization = session.headers.get("Authorization", "")
    token = authorization.removeprefix("Bearer ").strip()
    return text.replace(token, "[REDACTED]") if token else text


def parse_sse(response: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    event = "message"
    data_lines: list[str] = []
    for raw in response.iter_lines(decode_unicode=True):
        line = raw or ""
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif not line and data_lines:
            yield event, json.loads("\n".join(data_lines))
            event = "message"
            data_lines = []
    if data_lines:
        yield event, json.loads("\n".join(data_lines))


def run_stream(session: requests.Session, url: str, payload: dict[str, Any], case_id: str) -> tuple[Sample, dict[str, Any], str]:
    started = time.monotonic()
    first_token = None
    meta: dict[str, Any] = {}
    done: dict[str, Any] = {}
    text_parts: list[str] = []
    try:
        response = session.post(url, json=payload, stream=True, timeout=(20, 3600))
        response.raise_for_status()
        for event, data in parse_sse(response):
            if event == "meta":
                meta = data
            elif event == "token":
                if first_token is None:
                    first_token = time.monotonic()
                text_parts.append(str(data.get("text", "")))
            elif event == "error":
                raise RuntimeError(str(data.get("message") or data.get("error") or "stream error"))
            elif event == "done":
                done = data
        if not done:
            raise RuntimeError("stream ended without done event")
        sample = Sample(
            case_id=case_id,
            success=done.get("finish_reason") in {"stop", "length"},
            request_id=meta.get("request_id"),
            first_token_seconds=None if first_token is None else first_token - started,
            wall_seconds=time.monotonic() - started,
            completion_tokens=done.get("completion_tokens"),
            decode_tokens_per_second=done.get("decode_tokens_per_second"),
            finish_reason=done.get("finish_reason"),
            context_trimmed=meta.get("context_trimmed"),
        )
        return sample, meta, "".join(text_parts)
    except Exception as exc:
        error = scrub_session_secrets(str(exc), session)[:500]
        return Sample(case_id=case_id, success=False, error=error, wall_seconds=time.monotonic() - started), meta, "".join(text_parts)


def run_case(session: requests.Session, base_url: str, case: dict[str, Any]) -> Sample:
    payload = {"messages": case["messages"], "max_new_tokens": case["max_new_tokens"], "stream": True}
    sample, meta, text = run_stream(session, f"{base_url}/api/chat", payload, case["id"])
    sample.continuation_ok = None
    if sample.success and case["id"] == "long" and sample.finish_reason == "length":
        retained = meta.get("retained_messages")
        if not isinstance(retained, list) or not text:
            sample.continuation_ok = False
        else:
            continuation, _, continuation_text = run_stream(
                session,
                f"{base_url}/api/chat/continue",
                {"messages": retained, "prior_text": text, "max_new_tokens": case["max_new_tokens"], "stream": True},
                f"{case['id']}-continue",
            )
            sample.continuation_ok = continuation.success and bool(continuation_text.strip())
    return sample


def median(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(clean) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the deployed IngExuity CPU Space.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    session = build_session()

    runtime = session.get(f"{base_url}/api/runtime", timeout=60)
    runtime.raise_for_status()
    runtime_data = runtime.json()

    for case in CASES:
        run_case(session, base_url, case)  # warm-up, intentionally excluded

    samples: list[Sample] = []
    for case in CASES:
        for _ in range(3):
            samples.append(run_case(session, base_url, case))

    summary: dict[str, Any] = {}
    for case in CASES:
        successful = [sample for sample in samples if sample.case_id == case["id"] and sample.success]
        summary[case["id"]] = {
            "successful_samples": len(successful),
            "median_first_token_seconds": median([sample.first_token_seconds for sample in successful]),
            "median_wall_seconds": median([sample.wall_seconds for sample in successful]),
            "median_completion_tokens": median([sample.completion_tokens for sample in successful]),
            "median_decode_tokens_per_second": median([sample.decode_tokens_per_second for sample in successful]),
            "finish_reasons": [sample.finish_reason for sample in successful],
            "continuation_successes": sum(sample.continuation_ok is True for sample in successful),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "runtime": runtime_data,
        "cases": CASES,
        "samples": [asdict(sample) for sample in samples],
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    all_cases_green = all(summary[case["id"]]["successful_samples"] == 3 for case in CASES)
    long_samples = [sample for sample in samples if sample.case_id == "long" and sample.success]
    long_contract = bool(long_samples) and all(
        sample.finish_reason == "length" and (sample.completion_tokens or 0) >= 1000
        for sample in long_samples
    )
    continuation_contract = all(
        sample.finish_reason != "length" or sample.continuation_ok is True for sample in long_samples
    )
    return 0 if all_cases_green and long_contract and continuation_contract else 1


if __name__ == "__main__":
    raise SystemExit(main())
