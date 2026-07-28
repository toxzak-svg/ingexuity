import pytest

from spaces.hf_cpu.config import Settings
from spaces.hf_cpu.generation import (
    MessageTooLarge,
    build_continuation_messages,
    classify_finish_reason,
    conservative_message_token_upper_bound,
    normalize_request,
    parse_openai_sse,
)


def word_counter(messages):
    return sum(len(message["content"].split()) + 4 for message in messages)


def long_history():
    return [
        {"role": "user", "content": "old " * 1200},
        {"role": "assistant", "content": "reply " * 1200},
        {"role": "user", "content": "latest request"},
    ]


def test_default_and_hard_output_caps_are_explicit():
    request = normalize_request({"messages": [{"role": "user", "content": "x"}]})
    assert request.max_new_tokens == 1024
    assert normalize_request({"messages": [{"role": "user", "content": "x"}], "max_new_tokens": 2048}).max_new_tokens == 2048


def test_request_over_hard_cap_is_rejected():
    with pytest.raises(ValueError, match="max_new_tokens must be between 1 and 2048"):
        normalize_request({"messages": [{"role": "user", "content": "x"}], "max_new_tokens": 2049})


def test_context_trim_preserves_newest_user_turn_and_reports_it():
    request = normalize_request(
        {"messages": long_history(), "max_new_tokens": 1024},
        settings=Settings(context_tokens=3000),
        token_count=word_counter,
    )
    assert request.context_trimmed is True
    assert request.messages[-1] == {"role": "user", "content": "latest request"}


def test_context_trim_removes_complete_oldest_turn_and_keeps_system():
    settings = Settings(context_tokens=100, max_new_tokens=30, default_max_new_tokens=20, reserve_tokens=10)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "answer " * 30},
        {"role": "user", "content": "new request"},
    ]
    request = normalize_request(
        {"messages": messages, "max_new_tokens": 20},
        settings=settings,
        token_count=word_counter,
    )
    assert request.messages == [messages[0], messages[-1]]
    assert request.context_trimmed is True


def test_single_newest_message_over_budget_is_explicit():
    settings = Settings(context_tokens=100, max_new_tokens=30, default_max_new_tokens=20, reserve_tokens=10)
    with pytest.raises(MessageTooLarge) as exc:
        normalize_request(
            {"messages": [{"role": "user", "content": "x " * 100}], "max_new_tokens": 20},
            settings=settings,
            token_count=word_counter,
        )
    assert exc.value.allowed_tokens == 70


def test_conservative_fast_path_skips_exact_tokenizer_for_small_prompts():
    calls = 0

    def exact_counter(messages):
        nonlocal calls
        calls += 1
        raise AssertionError("exact tokenizer should not run")

    request = normalize_request(
        {"messages": [{"role": "user", "content": "Hello"}]},
        token_count=exact_counter,
        token_upper_bound=conservative_message_token_upper_bound,
    )
    assert request.context_trimmed is False
    assert calls == 0


def test_exact_token_counts_are_memoized_within_a_request():
    calls = 0

    def exact_counter(messages):
        nonlocal calls
        calls += 1
        return 10

    normalize_request(
        {"messages": [{"role": "user", "content": "Hello"}]},
        token_count=exact_counter,
    )
    assert calls == 1


def test_continuation_adds_a_non_repeating_instruction():
    messages = build_continuation_messages(
        [{"role": "user", "content": "Write an essay."}], "First section"
    )
    assert messages[-2] == {"role": "assistant", "content": "First section"}
    assert messages[-1]["content"] == "Continue exactly where the prior answer stopped. Do not repeat it."


def test_finish_reason_only_promotes_length():
    assert classify_finish_reason("length") == "length"
    assert classify_finish_reason("stop") == "stop"
    assert classify_finish_reason(None) == "stop"


def test_openai_sse_parser_ignores_keepalives_and_done():
    lines = [": keepalive", "", 'data: {"choices":[]}', "data: [DONE]"]
    assert list(parse_openai_sse(lines)) == [{"choices": []}]
