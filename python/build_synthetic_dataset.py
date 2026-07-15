#!/usr/bin/env python3
"""Build a deterministic synthetic IngExuity prediction dataset."""

from __future__ import annotations

import argparse
import json

from ingexuity_data.build import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_pilot")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--renderer", choices=["template"], default="template")
    args = parser.parse_args()
    manifest = build_dataset(args.output, count=args.count, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["accepted"] != manifest["requested"] or manifest["rejected"]:
        return 2
    if any(manifest["family_overlap"].values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
