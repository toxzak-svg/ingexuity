#!/usr/bin/env python3
"""Conversation-aware QLoRA/SFT training for IngExuity.

This script accepts JSON or JSONL records in any of these shapes:

    {"messages": [{"role": "user", "content": "..."}, ...]}
    {"prompt": "...", "completion": "..."}
    {"text": "..."}

Conversational records are expanded so that every assistant turn becomes a
prompt/completion example. TRL therefore computes loss only on the assistant
completion rather than teaching the model to imitate user messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from training_data import (
    DatasetFormatError,
    build_prompt_completion_examples,
    read_records,
)


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT = "./llama3_finetuned"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an instruct model on IngExuity conversations with QLoRA"
    )
    parser.add_argument("--data", required=True, help="JSON or JSONL training dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Training output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token; defaults to HF_TOKEN")

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataset-num-proc", type=int, default=None)

    parser.add_argument(
        "--packing",
        action="store_true",
        help="Pack examples into fixed-length sequences; best on modern GPUs",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip malformed records")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help="Train on at most N normalized examples for pipeline verification",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch <= 0 or args.eval_batch <= 0 or args.grad_accum <= 0:
        raise ValueError("Batch sizes and gradient accumulation must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    for name in ("validation_fraction", "test_fraction"):
        value = getattr(args, name)
        if value < 0 or value >= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        raise ValueError("Evaluation and save steps must be positive")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps")


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} ({props.total_memory / 1024**3:.1f} GiB VRAM)"


def choose_precision() -> tuple[torch.dtype, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def split_examples(
    examples: list[dict[str, str]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset | None, Dataset | None, dict[str, int]]:
    """Deterministically split examples while guaranteeing a training set."""

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)

    def requested_count(fraction: float) -> int:
        if fraction <= 0:
            return 0
        return max(1, int(round(len(indices) * fraction)))

    validation_count = requested_count(validation_fraction)
    test_count = requested_count(test_fraction)
    if validation_count + test_count >= len(indices):
        raise ValueError(
            "Dataset is too small for the requested validation/test fractions; "
            "reduce the fractions or provide more examples"
        )

    test_indices = indices[:test_count]
    validation_indices = indices[test_count : test_count + validation_count]
    train_indices = indices[test_count + validation_count :]

    def make_dataset(selected: list[int]) -> Dataset | None:
        if not selected:
            return None
        return Dataset.from_list([examples[index] for index in selected])

    counts = {
        "train": len(train_indices),
        "validation": len(validation_indices),
        "test": len(test_indices),
    }
    return (
        make_dataset(train_indices),
        make_dataset(validation_indices),
        make_dataset(test_indices),
        counts,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> Path:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"[Train] Device: {gpu_summary()}")
    print(f"[Train] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer has no EOS token; specify a compatible instruct model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = read_records(args.data)
    try:
        examples, data_stats = build_prompt_completion_examples(
            records,
            tokenizer,
            strict=not args.skip_invalid,
            deduplicate=not args.no_deduplicate,
        )
    except DatasetFormatError as exc:
        raise SystemExit(f"[Train] Dataset error: {exc}") from exc

    if args.smoke_test:
        examples = examples[: args.smoke_test]
        data_stats.output_examples = len(examples)
        print(f"[Train] Smoke-test mode: limited to {len(examples)} examples")

    train_dataset, validation_dataset, test_dataset, split_counts = split_examples(
        examples,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print(
        "[Train] Dataset: "
        f"{data_stats.input_records} records -> {len(examples)} examples; "
        f"train={split_counts['train']}, validation={split_counts['validation']}, "
        f"test={split_counts['test']}"
    )

    compute_dtype, use_bf16, use_fp16 = choose_precision()
    use_4bit = torch.cuda.is_available() and not args.no_4bit
    if not torch.cuda.is_available() and not args.no_4bit:
        print("[Train] CUDA unavailable; disabling 4-bit bitsandbytes loading")

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    effective_batch = args.batch * args.grad_accum * max(1, torch.cuda.device_count())
    approximate_steps = math.ceil(len(train_dataset) / effective_batch * args.epochs)
    has_validation = validation_dataset is not None
    checkpoint_strategy = (
        "steps"
        if approximate_steps >= max(args.eval_steps, args.save_steps)
        else "epoch"
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=(
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8
        ),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=checkpoint_strategy if has_validation else "no",
        eval_steps=(
            args.eval_steps
            if has_validation and checkpoint_strategy == "steps"
            else None
        ),
        save_strategy=checkpoint_strategy,
        save_steps=args.save_steps if checkpoint_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        packing=args.packing,
        eval_packing=False,
        completion_only_loss=True,
        dataset_num_proc=args.dataset_num_proc,
        model_init_kwargs={
            "dtype": compute_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
            **({"token": hf_token} if hf_token else {}),
        },
    )

    print(f"[Train] Effective batch size: {effective_batch}")
    print(f"[Train] Approximate optimizer steps: {approximate_steps}")
    print(f"[Train] Checkpoint strategy: {checkpoint_strategy}")
    print(f"[Train] Precision: {compute_dtype}; QLoRA 4-bit: {use_4bit}")

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        quantization_config=quantization_config,
    )
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    evaluation_metrics: dict[str, Any] = {}
    if validation_dataset is not None:
        validation_metrics = trainer.evaluate(
            eval_dataset=validation_dataset,
            metric_key_prefix="validation",
        )
        trainer.log_metrics("validation", validation_metrics)
        trainer.save_metrics("validation", validation_metrics)
        evaluation_metrics.update(validation_metrics)

    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        evaluation_metrics.update(test_metrics)

    manifest = {
        "model": args.model,
        "adapter_path": str(final_dir),
        "dataset_path": str(Path(args.data).resolve()),
        "dataset_sha256": sha256_file(args.data),
        "dataset_stats": data_stats.to_dict(),
        "split_counts": split_counts,
        "effective_batch_size": effective_batch,
        "approximate_optimizer_steps": approximate_steps,
        "compute_dtype": compute_dtype,
        "use_4bit": use_4bit,
        "evaluation_metrics": evaluation_metrics,
        "arguments": vars(args),
    }
    write_json(output_dir / "training_manifest.json", manifest)

    print(f"[Train] Adapter saved to: {final_dir}")
    print(f"[Train] Manifest saved to: {output_dir / 'training_manifest.json'}")
    return final_dir


if __name__ == "__main__":
    main()
