from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Callable, Iterable, Iterator, Mapping, Sequence

try:
    from .config import Settings
except ImportError:  # Direct execution inside the Space image.
    from config import Settings

ALLOWED_ROLES = {"system", "user", "assistant"}
CONTINUATION_INSTRUCTION = "Continue exactly where the prior answer stopped. Do not repeat it."


class MessageTooLarge(ValueError):
    def __init__(self, allowed_tokens: int) -> None:
        super().__init__(f"newest user message exceeds the {allowed_tokens}-token input budget")
        self.allowed_tokens = allowed_tokens


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict[str, str]]
    max_new_tokens: int
    context_trimmed: bool
    input_tokens: int
    input_budget: int


def _estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def estimate_message_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    return 3 + sum(4 + _estimate_text_tokens(message["content"]) for message in messages)


def conservative_message_token_upper_bound(messages: Sequence[Mapping[str, str]]) -> int:
    """Return a deliberately loose bound that is safe without calling the tokenizer.

    A tokenizer cannot emit more ordinary tokens than there are UTF-8 bytes in the
    message text. The per-message allowance and the runtime's separate template
    reserve cover role markers and chat-template framing.
    """
    return 3 + sum(8 + len(message["content"].encode("utf-8")) for message in messages)


def _validate_messages(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty array")
    if len(raw) > 32:
        raise ValueError("messages may contain at most 32 items")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"messages[{index}].role is invalid")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty text")
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("messages must contain a user turn")
    return messages


def _drop_oldest_turn(messages: list[dict[str, str]], protected_index: int) -> bool:
    start = 1 if messages and messages[0]["role"] == "system" else 0
    if start >= protected_index:
        return False
    end = start + 1
    while end < protected_index and messages[end]["role"] != "user":
        end += 1
    del messages[start:end]
    return True


def normalize_request(
    payload: Mapping[str, object],
    settings: Settings | None = None,
    token_count: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
    token_upper_bound: Callable[[Sequence[Mapping[str, str]]], int] | None = None,
) -> GenerationRequest:
    settings = settings or Settings()
    counter = token_count or estimate_message_tokens
    messages = _validate_messages(payload.get("messages"))
    max_new_tokens = payload.get("max_new_tokens", settings.default_max_new_tokens)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= settings.max_new_tokens:
        raise ValueError(f"max_new_tokens must be between 1 and {settings.max_new_tokens}")
    stream = payload.get("stream", True)
    if stream is not True:
        raise ValueError("stream must be true")

    input_budget = settings.context_tokens - max_new_tokens - settings.reserve_tokens
    retained = list(messages)

    # Most chat requests are far below the context ceiling. In that common case,
    # a byte-level upper bound proves the request fits and avoids repeated local
    # HTTP calls to llama-server's template and tokenizer endpoints.
    if token_upper_bound is not None and token_upper_bound(retained) <= input_budget:
        return GenerationRequest(
            messages=retained,
            max_new_tokens=max_new_tokens,
            context_trimmed=False,
            input_tokens=estimate_message_tokens(retained),
            input_budget=input_budget,
        )

    count_cache: dict[tuple[tuple[str, str], ...], int] = {}

    def exact_count(candidate: Sequence[Mapping[str, str]]) -> int:
        key = tuple((message["role"], message["content"]) for message in candidate)
        value = count_cache.get(key)
        if value is None:
            value = counter(candidate)
            count_cache[key] = value
        return value

    newest_user_index = max(index for index, message in enumerate(retained) if message["role"] == "user")
    newest_user = [retained[newest_user_index]]
    if exact_count(newest_user) > input_budget:
        raise MessageTooLarge(input_budget)

    context_trimmed = False
    while exact_count(retained) > input_budget:
        newest_user_index = max(index for index, message in enumerate(retained) if message["role"] == "user")
        if not _drop_oldest_turn(retained, newest_user_index):
            raise MessageTooLarge(input_budget)
        context_trimmed = True

    return GenerationRequest(
        messages=retained,
        max_new_tokens=max_new_tokens,
        context_trimmed=context_trimmed,
        input_tokens=exact_count(retained),
        input_budget=input_budget,
    )


def build_continuation_messages(messages: list[dict[str, str]], prior_text: str) -> list[dict[str, str]]:
    if not isinstance(prior_text, str) or not prior_text.strip():
        raise ValueError("prior_text must be non-empty text")
    return [
        *messages,
        {"role": "assistant", "content": prior_text},
        {"role": "user", "content": CONTINUATION_INSTRUCTION},
    ]


def classify_finish_reason(raw: str | None) -> str:
    return "length" if raw == "length" else "stop"


def sse(event: str, data: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n".encode("utf-8")


def parse_openai_sse(lines: Iterable[str | bytes]) -> Iterator[dict[str, object]]:
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        value = json.loads(data)
        if isinstance(value, dict):
            yield value


def extract_chunk(chunk: Mapping[str, object]) -> tuple[str, str | None, int | None, float | None]:
    text = ""
    finish_reason: str | None = None
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            text = delta["content"]
        raw_finish = choice.get("finish_reason")
        if isinstance(raw_finish, str):
            finish_reason = raw_finish
    completion_tokens: int | None = None
    usage = chunk.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        completion_tokens = usage["completion_tokens"]
    timings = chunk.get("timings")
    decode_rate: float | None = None
    if isinstance(timings, dict):
        if completion_tokens is None and isinstance(timings.get("predicted_n"), int):
            completion_tokens = timings["predicted_n"]
        rate = timings.get("predicted_per_second")
        if isinstance(rate, (int, float)):
            decode_rate = float(rate)
    return text, finish_reason, completion_tokens, decode_rate
