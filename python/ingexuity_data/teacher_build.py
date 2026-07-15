"""Resumable construction for teacher-rendered synthetic datasets."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .build import _split_families, _write_jsonl, sha256_file
from .curriculum import build_curriculum
from .render import TemplateRenderer
from .teacher import make_retry_prompt, make_teacher_prompt
from .validate import semantic_signature, validate_generated_record


def scale_gate(
    requested: int,
    accepted: int,
    duplicates: int,
    elapsed_seconds: float,
) -> dict:
    acceptance_rate = accepted / requested if requested else 0.0
    duplicate_rate = duplicates / requested if requested else 1.0
    estimated_hours = (elapsed_seconds / requested * 10_000 / 3600) if requested else float("inf")
    checks = {
        "acceptance_at_least_95_percent": acceptance_rate >= 0.95,
        "duplicates_at_most_2_percent": duplicate_rate <= 0.02,
        "estimated_10k_at_most_8_hours": estimated_hours <= 8.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "acceptance_rate": round(acceptance_rate, 6),
        "duplicate_rate": round(duplicate_rate, 6),
        "estimated_10k_hours": round(estimated_hours, 4),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _finalize(output: Path, accepted: list[dict], rejected: list[dict], count: int, seed: int) -> dict:
    families = sorted({record["family"] for record in accepted})
    split_families = _split_families(families, seed)
    family_to_split = {
        family: split
        for split, family_names in split_families.items()
        for family in family_names
    }
    splits: dict[str, list[dict]] = defaultdict(list)
    for record in accepted:
        splits[family_to_split[record["family"]]].append(record)
    for split in ("train", "eval", "test"):
        splits[split].sort(key=lambda row: row["scenario_id"])
        _write_jsonl(output / f"{split}.jsonl", splits[split])
    _write_jsonl(output / "quarantine.jsonl", rejected)
    overlap = {
        "train_eval": sorted(set(split_families["train"]) & set(split_families["eval"])),
        "train_test": sorted(set(split_families["train"]) & set(split_families["test"])),
        "eval_test": sorted(set(split_families["eval"]) & set(split_families["test"])),
    }
    files = {
        f"{name}.jsonl": sha256_file(output / f"{name}.jsonl")
        for name in ("train", "eval", "test", "quarantine")
    }
    manifest = {
        "dataset_version": "synthetic-prediction-teacher-v1",
        "schema_version": "1.0",
        "renderer": "local_teacher_surface_only",
        "seed": seed,
        "requested": count,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "complete": len(accepted) + len(rejected) == count,
        "split_counts": {split: len(splits[split]) for split in ("train", "eval", "test")},
        "split_families": split_families,
        "family_overlap": overlap,
        "category_counts": dict(sorted(Counter(row.get("curriculum", {}).get("category") for row in accepted).items())),
        "mode_counts": dict(sorted(Counter(row["response_mode"] for row in accepted).items())),
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Synthetic pilot data; model quality requires held-out structured evaluation.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_teacher_dataset(
    output_dir: str | Path,
    count: int,
    seed: int,
    teacher,
    stop_after: int | None = None,
) -> dict:
    """Generate teacher surfaces, checkpoint every row, and resume by scenario id."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "teacher-accepted.jsonl"
    rejected_path = output / "teacher-quarantine.jsonl"
    accepted = _read_jsonl(accepted_path)
    rejected = _read_jsonl(rejected_path)
    completed_ids = {row["scenario_id"] for row in accepted}
    completed_ids.update(row["scenario_id"] for row in rejected)
    signatures = {semantic_signature(row) for row in accepted}

    pending = [row for row in build_curriculum(count=count, seed=seed) if row["scenario_id"] not in completed_ids]
    if stop_after is not None:
        pending = pending[:stop_after]
    renderer = TemplateRenderer(seed=seed)
    drafts = [renderer.render(scenario) for scenario in pending]
    prompts = [make_teacher_prompt(scenario, draft) for scenario, draft in zip(pending, drafts)]
    started = time.monotonic()
    rewrites = teacher.rewrite_batch(prompts) if prompts else []
    if len(rewrites) != len(pending):
        raise ValueError("teacher returned a different number of rewrites than prompts")

    duplicates = 0
    duplicate_retries = 0
    for scenario, draft, initial_rewrite in zip(pending, drafts, rewrites):
        rewrite = initial_rewrite
        final_errors: list[str] = []
        accepted_record = None
        for attempt in range(3):
            record = None
            signature = None
            if isinstance(rewrite, Exception):
                errors = [str(rewrite)]
                previous_surface = None
            else:
                previous_surface = rewrite
                record = renderer.render_with_surface(scenario, rewrite)
                record["curriculum"] = {
                    key: scenario[key]
                    for key in (
                        "category", "domain", "communication_style", "time_horizon",
                        "evidence_level", "familiarity", "surface_seed",
                    )
                }
                record["teacher_attempts"] = attempt + 1
                errors = validate_generated_record(record)
                signature = semantic_signature(record)
                if signature in signatures:
                    errors.append("duplicate observable input and training target")
            if not errors:
                accepted_record = record
                break
            final_errors = errors
            duplicate_failure = "duplicate observable input and training target" in errors
            if attempt == 2:
                duplicates += int(duplicate_failure)
                break
            duplicate_retries += int(duplicate_failure)
            retry_results = teacher.rewrite_batch(
                [
                    make_retry_prompt(
                        scenario,
                        draft,
                        previous_surface,
                        "; ".join(errors),
                        attempt + 1,
                    )
                ]
            )
            if len(retry_results) != 1:
                raise ValueError("teacher retry returned a different number of rewrites than prompts")
            rewrite = retry_results[0]

        if accepted_record is None:
            entry = {"scenario_id": scenario["scenario_id"], "errors": final_errors}
            rejected.append(entry)
            _append_jsonl(rejected_path, entry)
            continue
        signatures.add(semantic_signature(accepted_record))
        accepted.append(accepted_record)
        _append_jsonl(accepted_path, accepted_record)

    elapsed = time.monotonic() - started
    complete = len(accepted) + len(rejected) == count
    if complete:
        manifest = _finalize(output, accepted, rejected, count, seed)
    else:
        manifest = {
            "requested": count,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "complete": False,
            "split_counts": {},
        }
    manifest["last_batch"] = {
        "attempted": len(pending),
        "duplicates": duplicates,
        "duplicate_retries": duplicate_retries,
        "elapsed_seconds": round(elapsed, 4),
    }
    return manifest
