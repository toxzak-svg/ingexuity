#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable

PUBLISH_ROOT_FILES = {
    ".hfignore",
    "Dockerfile",
    "NOTICE",
    "README.md",
    "app.py",
    "config.py",
    "generation.py",
    "requirements.txt",
    "runtime.py",
}
PUBLISH_DIRS = {"ui"}
BLOCKED_SUFFIXES = {".bin", ".gguf", ".safetensors"}
BLOCKED_PARTS = {"__pycache__", ".pytest_cache", ".cache", "tests"}
DELETE_PATTERNS = ["*", "**/*"]


def publishable_paths(source: Path) -> list[str]:
    files: list[str] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        parts = set(relative.parts)
        if parts & BLOCKED_PARTS or path.suffix.lower() in BLOCKED_SUFFIXES:
            continue
        posix = relative.as_posix()
        allowed = posix in PUBLISH_ROOT_FILES or relative.parts[0] in PUBLISH_DIRS
        if allowed:
            files.append(posix)
    return sorted(files)


def scrub_token(text: str, token: str) -> str:
    return text.replace(token, "[REDACTED]") if token else text


def stage_files(source: Path, destination: Path, paths: Iterable[str]) -> None:
    for relative in paths:
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the reviewed CPU Space source tree.")
    parser.add_argument("--repo-id", default="toxzak/ingexuity")
    parser.add_argument("--revision", required=True, help="Git commit SHA represented by this Space release")
    parser.add_argument("--source", type=Path, default=Path("spaces/hf_cpu"))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if len(args.revision) != 40 or any(character not in "0123456789abcdef" for character in args.revision.lower()):
        raise SystemExit("--revision must be a full 40-character git SHA")

    from huggingface_hub import HfApi

    paths = publishable_paths(args.source)
    if not paths:
        raise SystemExit("publish manifest is empty")
    with tempfile.TemporaryDirectory(prefix="ingexuity-hf-cpu-") as temp_dir:
        staged = Path(temp_dir)
        stage_files(args.source, staged, paths)
        api = HfApi(token=token)
        try:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="space",
                folder_path=staged,
                path_in_repo=".",
                commit_message=f"Deploy CPU Space from GitHub {args.revision}",
                ignore_patterns=["**/__pycache__/**", "**/.pytest_cache/**", "**/tests/**", "*.gguf", "*.safetensors", "*.bin"],
                delete_patterns=DELETE_PATTERNS,
            )
            deadline = time.monotonic() + args.timeout_seconds
            runtime = api.get_space_runtime(args.repo_id)
            while runtime.stage != "RUNNING" and time.monotonic() < deadline:
                if runtime.stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
                    raise RuntimeError(f"Space entered failure stage {runtime.stage}")
                time.sleep(10)
                runtime = api.get_space_runtime(args.repo_id)
            if runtime.stage != "RUNNING":
                raise TimeoutError(f"Space did not reach RUNNING; last stage was {runtime.stage}")
            print(f"Hub commit: {commit.oid}")
            print(f"Space stage: {runtime.stage}")
            print(f"Hardware: {runtime.hardware}")
        except Exception as exc:
            raise SystemExit(scrub_token(str(exc), token)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
