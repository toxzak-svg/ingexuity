from pathlib import Path


def test_dockerfile_is_cpu_only_pinned_and_static():
    source = Path("spaces/hf_cpu/Dockerfile").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("torch", "cuda", "peft", "bitsandbytes"):
        assert forbidden not in lowered
    assert "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3" in source
    assert "421677676671db8d33e216c039ed1b7e31b5d711" in source
    assert "-DBUILD_SHARED_LIBS=OFF" in source
    assert "--mount=type=secret,id=HF_TOKEN" in source
    assert "-G Ninja" in source
    assert "COPY --from=model-download /model/${MODEL_FILE} /opt/models/model.gguf" in source


def test_space_card_declares_docker_port_and_llama_attribution():
    source = Path("spaces/hf_cpu/README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in source
    assert "app_port: 8000" in source
    assert "Built with Llama" in source
