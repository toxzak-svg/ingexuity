#!/usr/bin/env python3
"""Build a deterministic synthetic IngExuity prediction dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingexuity_data.build import build_dataset
from ingexuity_data.teacher import LocalTeacher
from ingexuity_data.teacher_build import build_teacher_dataset, scale_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_pilot")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--renderer", choices=["template", "teacher"], default="template")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--benchmark-count", type=int, default=200)
    parser.add_argument("--require-scale-gate", action="store_true")
    args = parser.parse_args()
    if args.renderer == "template":
        manifest = build_dataset(args.output, count=args.count, seed=args.seed)
    else:
        output = Path(args.output)
        teacher = LocalTeacher(
            model_name=args.teacher_model,
            batch_size=args.teacher_batch_size,
        )
        if args.benchmark_count:
            benchmark_dir = output / "benchmark"
            gate_path = benchmark_dir / "scale-gate.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
            else:
                benchmark = build_teacher_dataset(
                    benchmark_dir,
                    count=args.benchmark_count,
                    seed=args.seed,
                    teacher=teacher,
                )
                gate = scale_gate(
                    requested=benchmark["requested"],
                    accepted=benchmark["accepted"],
                    duplicates=benchmark["last_batch"]["duplicates"],
                    elapsed_seconds=benchmark["last_batch"]["elapsed_seconds"],
                )
                gate_path.write_text(
                    json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            print(json.dumps({"teacher_scale_gate": gate}, indent=2, sort_keys=True))
            if args.require_scale_gate and not gate["passed"]:
                return 4
        manifest = build_teacher_dataset(
            output,
            count=args.count,
            seed=args.seed,
            teacher=teacher,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["accepted"] != manifest["requested"] or manifest["rejected"]:
        return 2
    if any(manifest["family_overlap"].values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
