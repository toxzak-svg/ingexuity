#!/usr/bin/env python3
"""Merge an IngExuity LoRA adapter and export a quantized GGUF model.

Requires a local llama.cpp checkout/build. Set ``LLAMA_CPP_DIR`` or pass
``--llama-cpp-dir``. The script performs all three stages:

1. Load the base model and merge the PEFT adapter.
2. Run llama.cpp's ``convert_hf_to_gguf.py`` to produce an F16 GGUF.
3. Run ``llama-quantize`` for Q4_K_M/Q5_K_M/Q8_0 output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ADAPTER = "./llama3_finetuned/final"
DEFAULT_OUTPUT = "./llama3-finetuned.Q4_K_M.gguf"
QUANT_TYPES = ("F16", "Q4_K_M", "Q5_K_M", "Q8_0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter and export it to GGUF")
    parser.add_argument("--model-path", default=DEFAULT_ADAPTER, help="PEFT adapter or merged HF model")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model id/path; defaults to adapter_config.json",
    )
    parser.add_argument("--merged-dir", default="./merged_hf", help="Directory for merged HF weights")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Final GGUF output path")
    parser.add_argument("--quant-type", default="Q4_K_M", choices=QUANT_TYPES)
    parser.add_argument(
        "--llama-cpp-dir",
        default=os.environ.get("LLAMA_CPP_DIR"),
        help="llama.cpp checkout/build directory; defaults to LLAMA_CPP_DIR",
    )
    parser.add_argument("--hf-token", default=None, help="Defaults to HF_TOKEN")
    parser.add_argument("--keep-f16", action="store_true", help="Keep the intermediate F16 GGUF")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories/files")
    return parser.parse_args()


def load_adapter_config(model_path: Path) -> dict:
    config_path = model_path / "adapter_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid adapter config: {config_path}")
    return value


def prepare_output_path(path: Path, *, force: bool, directory: bool = False) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"Output already exists: {path}; pass --force to replace it")
    if path.is_dir() or directory:
        shutil.rmtree(path)
    else:
        path.unlink()


def merge_model(
    model_path: Path,
    *,
    base_model: str | None,
    merged_dir: Path,
    hf_token: str | None,
    force: bool,
) -> tuple[Path, str]:
    """Return a merged HF model directory and the resolved base model id."""

    adapter_config = load_adapter_config(model_path)
    is_adapter = bool(adapter_config)

    if not is_adapter:
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"{model_path} is neither a PEFT adapter nor a Hugging Face model directory"
            )
        print(f"[Export] No adapter_config.json found; using merged model directly: {model_path}")
        resolved_base = base_model or str(model_path)
        return model_path, resolved_base

    resolved_base = base_model or adapter_config.get("base_model_name_or_path")
    if not resolved_base:
        raise RuntimeError("Could not determine the base model; pass --base-model")

    prepare_output_path(merged_dir, force=force, directory=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Loading base model: {resolved_base}")
    base = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        token=hf_token,
        dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    print(f"[Export] Loading adapter: {model_path}")
    peft_model = PeftModel.from_pretrained(base, str(model_path), is_trainable=False)
    print("[Export] Merging adapter into base weights")
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer_source = model_path if (model_path / "tokenizer_config.json").exists() else resolved_base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=hf_token, use_fast=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"[Export] Merged Hugging Face model saved to: {merged_dir}")
    return merged_dir, str(resolved_base)


def find_converter(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find convert_hf_to_gguf.py under {llama_cpp_dir}. "
        "Use a current llama.cpp checkout."
    )


def find_quantizer(llama_cpp_dir: Path) -> Path:
    names = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")
    directories = (
        llama_cpp_dir / "build" / "bin",
        llama_cpp_dir / "bin",
        llama_cpp_dir,
    )
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find llama-quantize under {llama_cpp_dir}. Build llama.cpp first."
    )


def run(command: list[str]) -> None:
    print("[Export] $ " + " ".join(command))
    subprocess.run(command, check=True)


def convert_to_gguf(
    merged_dir: Path,
    *,
    llama_cpp_dir: Path,
    output_path: Path,
    quant_type: str,
    keep_f16: bool,
    force: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_output_path(output_path, force=force)

    converter = find_converter(llama_cpp_dir)
    f16_path = output_path if quant_type == "F16" else output_path.with_suffix(".f16.gguf")
    if f16_path != output_path:
        prepare_output_path(f16_path, force=force)

    run(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )

    if quant_type != "F16":
        quantizer = find_quantizer(llama_cpp_dir)
        run([str(quantizer), str(f16_path), str(output_path), quant_type])
        if not keep_f16:
            f16_path.unlink(missing_ok=True)

    if not output_path.exists():
        raise RuntimeError(f"GGUF conversion completed without creating {output_path}")
    return output_path


def main() -> Path:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not args.llama_cpp_dir:
        raise SystemExit(
            "[Export] Set LLAMA_CPP_DIR or pass --llama-cpp-dir with a current llama.cpp checkout"
        )
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()
    if not llama_cpp_dir.exists():
        raise FileNotFoundError(f"llama.cpp directory does not exist: {llama_cpp_dir}")

    merged_model_dir, resolved_base = merge_model(
        model_path,
        base_model=args.base_model,
        merged_dir=merged_dir,
        hf_token=hf_token,
        force=args.force,
    )
    result = convert_to_gguf(
        merged_model_dir,
        llama_cpp_dir=llama_cpp_dir,
        output_path=output_path,
        quant_type=args.quant_type,
        keep_f16=args.keep_f16,
        force=args.force,
    )

    metadata = {
        "adapter_or_model_path": str(model_path),
        "base_model": resolved_base,
        "merged_model_dir": str(merged_model_dir),
        "gguf_path": str(result),
        "quant_type": args.quant_type,
    }
    metadata_path = result.with_suffix(result.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[Export] GGUF ready: {result}")
    print(f"[Export] Metadata: {metadata_path}")
    return result


if __name__ == "__main__":
    main()
