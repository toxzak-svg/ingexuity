from __future__ import annotations

from http import HTTPStatus
import json
import mimetypes
from pathlib import Path
import threading
import time
import uuid
from typing import Callable, Iterable, Iterator, Mapping
from urllib.parse import unquote
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server
from socketserver import ThreadingMixIn

import requests

try:
    from .config import LLAMA_CPP_REVISION, MODEL_FILE, MODEL_REPO, MODEL_REVISION, Settings
    from .generation import (
        MessageTooLarge,
        build_continuation_messages,
        classify_finish_reason,
        extract_chunk,
        normalize_request,
        parse_openai_sse,
        sse,
    )
    from .runtime import RuntimeManager
except ImportError:  # Direct execution inside the Space image.
    from config import LLAMA_CPP_REVISION, MODEL_FILE, MODEL_REPO, MODEL_REVISION, Settings
    from generation import (
        MessageTooLarge,
        build_continuation_messages,
        classify_finish_reason,
        extract_chunk,
        normalize_request,
        parse_openai_sse,
        sse,
    )
    from runtime import RuntimeManager

UI_ROOT = Path(__file__).with_name("ui")


class RuntimeUnavailable(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def is_ready(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        try:
            templated = self.session.post(
                f"{self.base_url}/apply-template",
                json={"messages": messages},
                timeout=20,
            )
            templated.raise_for_status()
            prompt = templated.json()["prompt"]
            tokenized = self.session.post(
                f"{self.base_url}/tokenize",
                json={"content": prompt, "add_special": True},
                timeout=20,
            )
            tokenized.raise_for_status()
            tokens = tokenized.json()["tokens"]
            return len(tokens)
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise RuntimeUnavailable("tokenizer is unavailable") from exc

    def stream(self, payload: Mapping[str, object]) -> Iterator[dict[str, object]]:
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=dict(payload),
                stream=True,
                timeout=(10, 3600),
            )
            response.raise_for_status()
            yield from parse_openai_sse(response.iter_lines(decode_unicode=True))
        except requests.RequestException as exc:
            raise RuntimeUnavailable("generation runtime is unavailable") from exc


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _status_line(status: int) -> str:
    return f"{status} {HTTPStatus(status).phrase}"


def _read_json(environ: Mapping[str, object]) -> dict[str, object]:
    raw_length = environ.get("CONTENT_LENGTH") or "0"
    try:
        length = int(str(raw_length))
    except ValueError as exc:
        raise ValueError("invalid content length") from exc
    if length <= 0 or length > 2_000_000:
        raise ValueError("request body must contain JSON")
    stream = environ["wsgi.input"]
    body = stream.read(length)
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def create_app(
    settings: Settings | None = None,
    llama_client: LlamaClient | object | None = None,
    runtime_snapshot: Callable[[], Mapping[str, object]] | None = None,
) -> Callable:
    settings = settings or Settings()
    llama_client = llama_client or LlamaClient(f"http://127.0.0.1:{settings.llama_port}")
    runtime_snapshot = runtime_snapshot or (lambda: {"state": "ready" if llama_client.is_ready() else "unavailable"})
    generation_lock = threading.Lock()
    timing_lock = threading.Lock()
    latest_timing: dict[str, object] = {}

    def json_response(start_response: Callable, status: int, payload: Mapping[str, object]) -> list[bytes]:
        body = _json_bytes(payload)
        start_response(_status_line(status), [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]

    def serve_static(path: str, start_response: Callable) -> list[bytes]:
        filename = "index.html" if path == "/" else unquote(path.lstrip("/"))
        if filename not in {"index.html", "app.js", "styles.css"}:
            return json_response(start_response, 404, {"error": "not_found"})
        file_path = UI_ROOT / filename
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        start_response(_status_line(200), [("Content-Type", f"{content_type}; charset=utf-8"), ("Content-Length", str(len(body)))])
        return [body]

    def stream_chat(payload: dict[str, object], continuation: bool) -> tuple[int, dict[str, object] | None, Iterable[bytes] | None]:
        nonlocal latest_timing
        snapshot = runtime_snapshot()
        if snapshot.get("state") != "ready" or not llama_client.is_ready():
            return 503, {"error": "runtime_unavailable"}, None
        if not generation_lock.acquire(blocking=False):
            return 429, {"error": "generation_in_progress"}, None
        try:
            if continuation:
                messages = payload.get("messages")
                prior_text = payload.get("prior_text")
                if not isinstance(messages, list):
                    raise ValueError("messages must be an array")
                payload = dict(payload)
                payload["messages"] = build_continuation_messages(messages, prior_text)
            request = normalize_request(payload, settings=settings, token_count=llama_client.count_messages)
        except MessageTooLarge as exc:
            generation_lock.release()
            return 422, {"error": "message_too_large", "allowed_input_tokens": exc.allowed_tokens}, None
        except RuntimeUnavailable:
            generation_lock.release()
            return 503, {"error": "runtime_unavailable"}, None
        except ValueError as exc:
            generation_lock.release()
            return 400, {"error": "invalid_request", "message": str(exc)}, None

        request_id = uuid.uuid4().hex
        started = time.monotonic()

        def events() -> Iterator[bytes]:
            completion_tokens = 0
            decode_rate: float | None = None
            raw_finish: str | None = None
            first_token_seconds: float | None = None
            try:
                yield sse("meta", {
                    "request_id": request_id,
                    "context_trimmed": request.context_trimmed,
                    "max_new_tokens": request.max_new_tokens,
                    "input_tokens": request.input_tokens,
                    "input_budget": request.input_budget,
                    "retained_messages": request.messages,
                })
                upstream = {
                    "messages": request.messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "cache_prompt": True,
                    "max_tokens": request.max_new_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                for chunk in llama_client.stream(upstream):
                    text, chunk_finish, chunk_tokens, chunk_rate = extract_chunk(chunk)
                    if text:
                        if first_token_seconds is None:
                            first_token_seconds = time.monotonic() - started
                        yield sse("token", {"text": text})
                    if chunk_finish is not None:
                        raw_finish = chunk_finish
                    if chunk_tokens is not None:
                        completion_tokens = chunk_tokens
                    if chunk_rate is not None:
                        decode_rate = chunk_rate
                finish_reason = classify_finish_reason(raw_finish)
                wall_seconds = time.monotonic() - started
                done = {
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "decode_tokens_per_second": decode_rate,
                    "can_continue": finish_reason == "length",
                }
                with timing_lock:
                    latest_timing = {
                        "request_id": request_id,
                        "first_token_seconds": first_token_seconds,
                        "wall_seconds": wall_seconds,
                        **done,
                    }
                yield sse("done", done)
            except Exception:
                yield sse("error", {"error": "runtime_unavailable", "message": "generation runtime failed"})
                yield sse("done", {"finish_reason": "error", "completion_tokens": completion_tokens, "decode_tokens_per_second": decode_rate, "can_continue": False})
            finally:
                generation_lock.release()

        return 200, None, events()

    def app(environ: Mapping[str, object], start_response: Callable) -> Iterable[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        if method == "GET" and path in {"/", "/index.html", "/app.js", "/styles.css"}:
            return serve_static(path, start_response)
        if method == "GET" and path == "/health":
            state = runtime_snapshot().get("state")
            status = 200 if state == "ready" else 503
            return json_response(start_response, status, {"status": "ok" if status == 200 else "loading", "state": state})
        if method == "GET" and path == "/api/runtime":
            snapshot = dict(runtime_snapshot())
            with timing_lock:
                timing = dict(latest_timing)
            snapshot.update({
                "model": f"{MODEL_REPO}:{MODEL_FILE}",
                "model_revision": MODEL_REVISION,
                "llama_cpp_revision": LLAMA_CPP_REVISION,
                "context_tokens": settings.context_tokens,
                "default_max_new_tokens": settings.default_max_new_tokens,
                "maximum_max_new_tokens": settings.max_new_tokens,
                "threads": settings.threads,
                "parallel": 1,
                "latest_timing": timing,
            })
            return json_response(start_response, 200, snapshot)
        if method == "POST" and path in {"/api/chat", "/api/chat/continue"}:
            try:
                payload = _read_json(environ)
            except ValueError as exc:
                return json_response(start_response, 400, {"error": "invalid_request", "message": str(exc)})
            status, error, body = stream_chat(payload, path.endswith("/continue"))
            if error is not None:
                return json_response(start_response, status, error)
            start_response(_status_line(200), [
                ("Content-Type", "text/event-stream; charset=utf-8"),
                ("Cache-Control", "no-cache, no-transform"),
                ("X-Accel-Buffering", "no"),
            ])
            return body or []
        return json_response(start_response, 404, {"error": "not_found"})

    return app


def main() -> None:
    settings = Settings()
    runtime = RuntimeManager(settings)
    runtime.start()
    runtime.wait_until_ready()
    llama_client = LlamaClient(runtime.base_url, runtime.session)
    app = create_app(settings=settings, llama_client=llama_client, runtime_snapshot=runtime.snapshot)
    server = make_server("0.0.0.0", settings.public_port, app, server_class=ThreadingWSGIServer, handler_class=WSGIRequestHandler)
    try:
        server.serve_forever()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
