# syntax=docker/dockerfile:1.7
FROM debian:bookworm AS llama-build
ARG LLAMA_CPP_REVISION=aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      libgomp1 \
      ninja-build \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /src/llama.cpp \
 && cd /src/llama.cpp \
 && git checkout "$LLAMA_CPP_REVISION"
RUN cmake -G Ninja -S /src/llama.cpp -B /src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_OPENMP=ON \
      -DGGML_NATIVE=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_BACKEND_DL=OFF \
      -DLLAMA_CURL=OFF \
 && cmake --build /src/build --target llama-server -j 2

FROM python:3.12-slim AS model-download
ARG MODEL_REPO=bartowski/Llama-3.2-1B-Instruct-GGUF
ARG MODEL_FILE=Llama-3.2-1B-Instruct-Q4_K_M.gguf
ARG MODEL_REVISION=421677676671db8d33e216c039ed1b7e31b5d711
ENV MODEL_REPO=${MODEL_REPO} MODEL_FILE=${MODEL_FILE} MODEL_REVISION=${MODEL_REVISION}
RUN pip install --no-cache-dir huggingface_hub==0.34.4
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
repo = os.environ.get("MODEL_REPO", "bartowski/Llama-3.2-1B-Instruct-GGUF")
filename = os.environ.get("MODEL_FILE", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
revision = os.environ.get("MODEL_REVISION", "421677676671db8d33e216c039ed1b7e31b5d711")
token_path = Path("/run/secrets/HF_TOKEN")
token = token_path.read_text().strip() if token_path.exists() else None
path = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision, token=token, local_dir="/model"))
if not path.is_file() or path.stat().st_size < 100_000_000:
    raise RuntimeError("downloaded GGUF is missing or unexpectedly small")
PY

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user
COPY --from=llama-build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=model-download /model/Llama-3.2-1B-Instruct-Q4_K_M.gguf /opt/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
WORKDIR /home/user/app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=user:user . /home/user/app
USER user
ENV PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py"]
