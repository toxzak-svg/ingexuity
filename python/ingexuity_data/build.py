"""Build versioned synthetic datasets with family-aware splits and manifests."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .render import TemplateRenderer
from .scenarios import generate_scenarios
from .validate import semantic_signature, validate_generated_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _split_families(families: list[str], seed: int) -> dict[str, list[str]]:
    if len(families) < 3:
        raise ValueError("family-aware splitting requires at least three scenario families")
    shuffled = sorted(families)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.2))
    eval_count = max(1, round(len(shuffled) * 0.2))
    return {
        "test": sorted(shuffled[:test_count]),
        "eval": sorted(shuffled[test_count : test_count + eval_count]),
        "train": sorted(shuffled[test_count + eval_count :]),
    }


def build_dataset(output_dir: str | Path, count: int = 100, seed: int = 42) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    renderer = TemplateRenderer(seed=seed)
    accepted: list[dict] = []
    quarantined: list[dict] = []
    signatures: set[str] = set()

    for scenario in generate_scenarios(count=count, seed=seed):
        record = renderer.render(scenario)
        errors = validate_generated_record(record)
        signature = semantic_signature(record)
        if signature in signatures:
            errors.append("duplicate observable input and training target")
        if errors:
            quarantined.append(
                {"scenario_id": scenario["scenario_id"], "errors": errors, "record": record}
            )
            continue
        signatures.add(signature)
        accepted.append(record)

    families = sorted({record["family"] for record in accepted})
    split_families = _split_families(families, seed)
    family_to_split = {
        family: split
        for split, split_family_names in split_families.items()
        for family in split_family_names
    }
    splits: dict[str, list[dict]] = defaultdict(list)
    for record in accepted:
        splits[family_to_split[record["family"]]].append(record)
    for split in ("train", "eval", "test"):
        splits[split].sort(key=lambda row: row["scenario_id"])
        _write_jsonl(output / f"{split}.jsonl", splits[split])
    _write_jsonl(output / "quarantine.jsonl", quarantined)

    overlap = {
        "train_eval": sorted(set(split_families["train"]) & set(split_families["eval"])),
        "train_test": sorted(set(split_families["train"]) & set(split_families["test"])),
        "eval_test": sorted(set(split_families["eval"]) & set(split_families["test"])),
    }
    files = {
        f"{split}.jsonl": sha256_file(output / f"{split}.jsonl")
        for split in ("train", "eval", "test", "quarantine")
    }
    manifest = {
        "dataset_version": "synthetic-prediction-pilot-v1",
        "schema_version": "1.0",
        "renderer": "template",
        "seed": seed,
        "requested": count,
        "accepted": len(accepted),
        "rejected": len(quarantined),
        "split_counts": {split: len(splits[split]) for split in ("train", "eval", "test")},
        "split_families": split_families,
        "family_overlap": overlap,
        "mode_counts": dict(sorted(Counter(row["response_mode"] for row in accepted).items())),
        "kind_counts": dict(sorted(Counter(row["scenario_kind"] for row in accepted).items())),
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Pipeline smoke only; this dataset does not establish model-quality improvement.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
