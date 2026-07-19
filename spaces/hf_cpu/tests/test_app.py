from io import BytesIO
import json

from spaces.hf_cpu.app import create_app
from spaces.hf_cpu.config import Settings
from spaces.hf_cpu.generation import estimate_message_tokens


class FakeLlama:
    def __init__(self):
        self.chunks = []
        self.ready = True
        self.payloads = []

    def is_ready(self):
        return self.ready

    def count_messages(self, messages):
        return estimate_message_tokens(messages)

    def stream(self, payload):
        self.payloads.append(payload)
        yield from self.chunks


def request(app, path, payload=None, method="POST"):
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(data)),
        "CONTENT_TYPE": "application/json",
        "wsgi.input": BytesIO(data),
    }
    result = {}

    def start_response(status, headers):
        result["status"] = int(status.split()[0])
        result["headers"] = dict(headers)

    body = b"".join(app(environ, start_response))
    result["body"] = body
    result["text"] = body.decode("utf-8")
    return result


def make_app(fake):
    return create_app(
        settings=Settings(),
        llama_client=fake,
        runtime_snapshot=lambda: {"state": "ready", "exit_code": None, "logs": ""},
    )


def test_chat_streams_meta_tokens_and_length_done():
    fake = FakeLlama()
    fake.chunks = [
        {"choices": [{"delta": {"content": "Part one"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}], "timings": {"predicted_n": 1024, "predicted_per_second": 7.5}},
    ]
    response = request(make_app(fake), "/api/chat", {"messages": [{"role": "user", "content": "Write a long answer"}], "stream": True})
    assert response["status"] == 200
    assert "event: meta" in response["text"]
    assert '"text":"Part one"' in response["text"]
    assert '"finish_reason":"length"' in response["text"]
    assert '"can_continue":true' in response["text"]
    assert fake.payloads[0]["max_tokens"] == 1024
    assert fake.payloads[0]["cache_prompt"] is True


def test_continue_is_not_offered_after_eos():
    fake = FakeLlama()
    fake.chunks = [{"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}]
    response = request(make_app(fake), "/api/chat", {"messages": [{"role": "user", "content": "Hello"}], "stream": True})
    assert '"can_continue":false' in response["text"]


def test_continue_builds_non_repeating_followup():
    fake = FakeLlama()
    fake.chunks = [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    response = request(make_app(fake), "/api/chat/continue", {
        "messages": [{"role": "user", "content": "Write an essay"}],
        "prior_text": "First part",
        "max_new_tokens": 512,
        "stream": True,
    })
    assert response["status"] == 200
    upstream_messages = fake.payloads[0]["messages"]
    assert upstream_messages[-2] == {"role": "assistant", "content": "First part"}
    assert upstream_messages[-1]["content"].startswith("Continue exactly")


def test_over_budget_message_is_a_clear_422():
    fake = FakeLlama()
    response = request(make_app(fake), "/api/chat", {"messages": [{"role": "user", "content": "x" * 100000}], "stream": True})
    assert response["status"] == 422
    assert json.loads(response["body"])["error"] == "message_too_large"


def test_failed_runtime_is_503_before_sse_headers():
    fake = FakeLlama()
    fake.ready = False
    app = create_app(
        settings=Settings(),
        llama_client=fake,
        runtime_snapshot=lambda: {"state": "failed"},
    )
    response = request(app, "/api/chat", {"messages": [{"role": "user", "content": "Hello"}], "stream": True})
    assert response["status"] == 503
    assert response["headers"]["Content-Type"] == "application/json"


def test_runtime_endpoint_is_truthful_and_prompt_free():
    fake = FakeLlama()
    response = request(make_app(fake), "/api/runtime", method="GET")
    data = json.loads(response["body"])
    assert data["context_tokens"] == 4096
    assert data["default_max_new_tokens"] == 1024
    assert data["maximum_max_new_tokens"] == 2048
    assert "messages" not in data
