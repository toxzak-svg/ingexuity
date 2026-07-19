from spaces.hf_cpu.config import Settings
from spaces.hf_cpu.runtime import build_llama_server_command


def test_cpu_basic_command_uses_two_threads_and_one_slot():
    command = build_llama_server_command(Settings())
    assert command[command.index("--threads") + 1] == "2"
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--parallel") + 1] == "1"
    assert "--flash-attn" not in command


def test_runtime_does_not_include_cuda_or_torch_arguments():
    command = " ".join(build_llama_server_command(Settings()))
    assert "cuda" not in command.lower()
    assert "torch" not in command.lower()
