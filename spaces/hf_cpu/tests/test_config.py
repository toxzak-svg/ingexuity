import pytest

from spaces.hf_cpu.config import MODEL_PATH, Settings


def test_model_uses_a_stable_runtime_path():
    assert MODEL_PATH == "/opt/models/model.gguf"


def test_invalid_integer_environment_names_the_setting(monkeypatch):
    monkeypatch.setenv("INGEXUITY_THREADS", "not-an-int")
    with pytest.raises(ValueError, match="INGEXUITY_THREADS.*not-an-int"):
        Settings()
