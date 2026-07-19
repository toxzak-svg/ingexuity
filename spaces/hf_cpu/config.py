from __future__ import annotations

from dataclasses import dataclass, field
import os

MODEL_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_REVISION = "421677676671db8d33e216c039ed1b7e31b5d711"
MODEL_PATH = "/opt/models/model.gguf"
LLAMA_CPP_REVISION = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for environment variable {name!r}: {value!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    context_tokens: int = field(default_factory=lambda: _env_int("INGEXUITY_CONTEXT_TOKENS", 4096))
    default_max_new_tokens: int = field(default_factory=lambda: _env_int("INGEXUITY_DEFAULT_MAX_NEW_TOKENS", 1024))
    max_new_tokens: int = field(default_factory=lambda: _env_int("INGEXUITY_MAX_NEW_TOKENS", 2048))
    threads: int = field(default_factory=lambda: _env_int("INGEXUITY_THREADS", 2))
    reserve_tokens: int = field(default_factory=lambda: _env_int("INGEXUITY_TEMPLATE_RESERVE_TOKENS", 160))
    public_port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    llama_port: int = field(default_factory=lambda: _env_int("INGEXUITY_LLAMA_PORT", 8081))
    startup_timeout_seconds: int = field(default_factory=lambda: _env_int("INGEXUITY_STARTUP_TIMEOUT_SECONDS", 600))

    def __post_init__(self) -> None:
        if not 1 <= self.default_max_new_tokens <= self.max_new_tokens:
            raise ValueError("default_max_new_tokens must be within the configured maximum")
        if self.context_tokens <= self.max_new_tokens + self.reserve_tokens:
            raise ValueError("context_tokens must leave input room after output and template reserve")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if not 1 <= self.public_port <= 65535 or not 1 <= self.llama_port <= 65535:
            raise ValueError("ports must be between 1 and 65535")
