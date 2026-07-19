from scripts.benchmark_hf_cpu import build_session, scrub_session_secrets


def test_private_space_session_uses_bearer_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_private_test_token")
    session = build_session()
    assert session.headers["Authorization"] == "Bearer hf_private_test_token"


def test_public_space_session_omits_authorization():
    session = build_session(token="")
    assert "Authorization" not in session.headers


def test_benchmark_errors_redact_the_token():
    session = build_session(token="hf_private_test_token")
    scrubbed = scrub_session_secrets(
        "request failed with hf_private_test_token",
        session,
    )
    assert scrubbed == "request failed with [REDACTED]"
