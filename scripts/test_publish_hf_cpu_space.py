from pathlib import Path

from scripts.publish_hf_cpu_space import DELETE_PATTERNS, publishable_paths, scrub_token


def test_publish_manifest_excludes_models_caches_and_training_artifacts(tmp_path):
    root = tmp_path / "space"
    (root / "ui").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".cache").mkdir()
    for name in ["Dockerfile", "app.py", "README.md", "NOTICE", ".hfignore"]:
        (root / name).write_text("ok", encoding="utf-8")
    (root / "ui" / "app.js").write_text("ok", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text("no", encoding="utf-8")
    (root / ".cache" / "blob").write_text("no", encoding="utf-8")
    (root / "model.gguf").write_text("no", encoding="utf-8")
    (root / "adapter_model.safetensors").write_text("no", encoding="utf-8")

    files = publishable_paths(root)
    assert "Dockerfile" in files
    assert "ui/app.js" in files
    assert "NOTICE" in files
    assert all(".cache" not in path for path in files)
    assert all("adapter_model" not in path for path in files)
    assert all(not path.endswith((".gguf", ".safetensors", ".bin")) for path in files)
    assert all(not path.startswith("tests/") for path in files)


def test_token_scrubber_removes_secret():
    assert scrub_token("failed hf_secret_value", "hf_secret_value") == "failed [REDACTED]"


def test_publish_replaces_unreviewed_remote_source():
    assert DELETE_PATTERNS == ["*", "**/*"]
