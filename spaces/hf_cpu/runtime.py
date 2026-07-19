from __future__ import annotations

from collections import deque
import atexit
import subprocess
import threading
import time
from typing import Any

import requests

try:
    from .config import MODEL_PATH, Settings
except ImportError:  # Direct execution inside the Space image.
    from config import MODEL_PATH, Settings


def build_llama_server_command(settings: Settings) -> list[str]:
    return [
        "llama-server",
        "--model", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(settings.llama_port),
        "--threads", str(settings.threads),
        "--ctx-size", str(settings.context_tokens),
        "--parallel", "1",
        "--cont-batching",
        "--cache-prompt",
        "--jinja",
        "--no-warmup",
        "--metrics",
        "--log-verbosity", "1",
    ]


class RuntimeManager:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.process: subprocess.Popen[str] | None = None
        self.state = "stopped"
        self.exit_code: int | None = None
        self._logs: deque[str] = deque(maxlen=120)
        self._lock = threading.Lock()
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.settings.llama_port}"

    def start(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                return
            self.state = "loading"
            self.exit_code = None
            command = build_llama_server_command(self.settings)
            # The executable and flags are constants or validated integers; no shell is involved.
            self.process = subprocess.Popen(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                close_fds=True,
            )
            threading.Thread(target=self._capture_logs, daemon=True).start()

    def _capture_logs(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            lowered = line.lower()
            if any(marker in lowered for marker in ('"messages"', '"content"', 'prompt:')):
                line = "[request content redacted]\n"
            self._logs.append(line[-500:])
        self.exit_code = process.poll()
        if self.state != "stopped":
            self.state = "failed"

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                self.exit_code = None if self.process is None else self.process.returncode
                self.state = "failed"
                raise RuntimeError(f"llama-server exited during startup with code {self.exit_code}")
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self._warm_up()
                    self.state = "ready"
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.state = "failed"
        raise TimeoutError("llama-server did not become ready before the startup timeout")

    def _warm_up(self) -> None:
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
                "cache_prompt": True,
            },
            timeout=120,
        )
        response.raise_for_status()

    def stop(self) -> None:
        process = self.process
        self.state = "stopped"
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if process:
            self.exit_code = process.poll()

    def snapshot(self) -> dict[str, Any]:
        process = self.process
        if process and process.poll() is not None and self.state not in {"stopped", "failed"}:
            self.state = "failed"
            self.exit_code = process.returncode
        logs = "".join(self._logs)[-2000:]
        return {"state": self.state, "exit_code": self.exit_code, "logs": logs}
