#!/usr/bin/env python3
"""Dataset normalization for IngExuity supervised fine-tuning.

The public entry point is ``build_prompt_completion_examples``. It accepts the
common JSON/JSONL formats used by exported chat datasets and converts every
assistant turn into a standard string prompt/completion example. This makes
completion-only loss reliable even when a model chat template does not expose
assistant-generation masks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROLE_ALIASES = {
    "human": "user",
    "person": "user",
    "bot": "assistant",
    "model": "assistant",
    "gpt": "assistant",
}
SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}


class DatasetFormatError(ValueError):
    """Raised when a training record cannot be normalized safely."""


@dataclass
class DatasetStats:
    input_records: int = 0
    output_examples: int = 0
    conversational_records: int = 0
    prompt_completion_records: int = 0
    plain_text_records: int = 0
    skipped_records: int = 0
    duplicates_removed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON or JSONL dataset into a list of mappings."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise DatasetFormatError(
                        f"Line {line_number} must contain a JSON object, got {type(value).__name__}"
                    )
                records.append(dict(value))
        return records

    if dataset_path.suffix.lower() != ".json":
        raise DatasetFormatError(
            f"Unsupported dataset extension {dataset_path.suffix!r}; use .json or .jsonl"
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if isinstance(value, list):
        records = value
    elif isinstance(value, Mapping):
        records = None
        for key in ("data", "records", "examples", "items"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            records = [value]
    else:
        raise DatasetFormatError(
            f"Top-level JSON must be an object or array, got {type(value).__name__}"
        )

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DatasetFormatError(
                f"Record {index} must be an object, got {type(record).__name__}"
            )
        normalized.append(dict(record))
    return normalized


def _text_content(value: Any, *, location: str) -> str:
    """Normalize string or OpenAI-style text content blocks to plain text."""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        pieces: list[str] = []
        for index, part in enumerate(value):
            if isinstance(part, str):
                pieces.append(part)
                continue
            if isinstance(part, Mapping):
                part_type = part.get("type")
                text = part.get("text")
                if part_type in (None, "text", "input_text", "output_text") and isinstance(text, str):
                    pieces.append(text)
                    continue
            raise DatasetFormatError(
                f"{location}[{index}] is not a supported text content block"
            )
        return "\n".join(piece.strip() for piece in pieces if piece.strip()).strip()

    raise DatasetFormatError(
        f"{location} must be a string or list of text blocks, got {type(value).__name__}"
    )


def normalize_messages(messages: Any, *, record_index: int) -> list[dict[str, str]]:
    """Validate and normalize a conversational message list."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise DatasetFormatError(f"Record {record_index}: 'messages' must be a list")

    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise DatasetFormatError(
                f"Record {record_index}, message {message_index}: expected an object"
            )

        role_value = message.get("role")
        if not isinstance(role_value, str):
            raise DatasetFormatError(
                f"Record {record_index}, message {message_index}: missing string role"
            )
        role = ROLE_ALIASES.get(role_value.strip().lower(), role_value.strip().lower())
        if role not in SUPPORTED_ROLES:
            raise DatasetFormatError(
                f"Record {record_index}, message {message_index}: unsupported role {role_value!r}"
            )

        content = _text_content(
            message.get("content"),
            location=f"Record {record_index}, message {message_index} content",
        )
        if not content:
            raise DatasetFormatError(
                f"Record {record_index}, message {message_index}: empty content"
            )

        normalized.append({"role": role, "content": content})

    if not normalized:
        raise DatasetFormatError(f"Record {record_index}: conversation is empty")
    return normalized


def _append_eos(text: str, eos_token: str | None) -> str:
    stripped = text.rstrip()
    if not eos_token or stripped.endswith(eos_token):
        return stripped
    return stripped + eos_token


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise DatasetFormatError("Tokenizer does not provide apply_chat_template")
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise DatasetFormatError(f"Could not apply model chat template: {exc}") from exc
    if not isinstance(rendered, str) or not rendered:
        raise DatasetFormatError("Model chat template returned an empty prompt")
    return rendered


def _completion_from_value(value: Any, *, record_index: int) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        messages = normalize_messages(value, record_index=record_index)
        assistant_parts = [m["content"] for m in messages if m["role"] == "assistant"]
        if not assistant_parts:
            raise DatasetFormatError(
                f"Record {record_index}: conversational completion has no assistant message"
            )
        return "\n".join(assistant_parts).strip()
    raise DatasetFormatError(
        f"Record {record_index}: completion must be a string or message list"
    )


def _prompt_from_value(tokenizer: Any, value: Any, *, record_index: int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        messages = normalize_messages(value, record_index=record_index)
        return _render_prompt(tokenizer, messages)
    raise DatasetFormatError(f"Record {record_index}: prompt must be a string or message list")


def iter_prompt_completion_examples(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    strict: bool = True,
    stats: DatasetStats | None = None,
) -> Iterator[dict[str, str]]:
    """Yield normalized string prompt/completion examples.

    For a multi-turn ``messages`` record, every assistant turn becomes its own
    example. The prompt contains all preceding messages, while the completion
    contains only that assistant response.
    """

    stats = stats or DatasetStats()
    eos_token = getattr(tokenizer, "eos_token", None)

    for record_index, raw_record in enumerate(records):
        stats.input_records += 1
        try:
            record = dict(raw_record)
            if "messages" in record:
                stats.conversational_records += 1
                messages = normalize_messages(record["messages"], record_index=record_index)
                produced = 0
                for message_index, message in enumerate(messages):
                    if message["role"] != "assistant":
                        continue
                    history = messages[:message_index]
                    if not history:
                        continue
                    yield {
                        "prompt": _render_prompt(tokenizer, history),
                        "completion": _append_eos(message["content"], eos_token),
                    }
                    produced += 1
                    stats.output_examples += 1
                if produced == 0:
                    raise DatasetFormatError(
                        f"Record {record_index}: conversation contains no trainable assistant turn"
                    )
                continue

            if "prompt" in record and "completion" in record:
                stats.prompt_completion_records += 1
                prompt = _prompt_from_value(tokenizer, record["prompt"], record_index=record_index)
                completion = _completion_from_value(record["completion"], record_index=record_index)
                if not completion:
                    raise DatasetFormatError(f"Record {record_index}: empty completion")
                yield {
                    "prompt": prompt,
                    "completion": _append_eos(completion, eos_token),
                }
                stats.output_examples += 1
                continue

            plain_value = record.get("text", record.get("content"))
            if plain_value is not None:
                stats.plain_text_records += 1
                text = _text_content(plain_value, location=f"Record {record_index} text")
                if not text:
                    raise DatasetFormatError(f"Record {record_index}: empty text")
                yield {"prompt": "", "completion": _append_eos(text, eos_token)}
                stats.output_examples += 1
                continue

            raise DatasetFormatError(
                f"Record {record_index}: expected messages, prompt/completion, text, or content"
            )
        except DatasetFormatError:
            if strict:
                raise
            stats.skipped_records += 1


def build_prompt_completion_examples(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    strict: bool = True,
    deduplicate: bool = True,
) -> tuple[list[dict[str, str]], DatasetStats]:
    """Normalize records and optionally remove exact duplicate examples."""

    stats = DatasetStats()
    examples = list(
        iter_prompt_completion_examples(records, tokenizer, strict=strict, stats=stats)
    )

    if deduplicate:
        unique: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for example in examples:
            key = (example["prompt"], example["completion"])
            if key in seen:
                stats.duplicates_removed += 1
                continue
            seen.add(key)
            unique.append(example)
        examples = unique
        stats.output_examples = len(examples)

    if not examples:
        raise DatasetFormatError("Dataset produced zero usable training examples")

    return examples, stats
