#!/bin/sh
# IngExuity startup script.
# Pre-warms the model BEFORE binding the HTTP port so Railway doesn't mark
# the deploy healthy before the LLM is actually ready to serve.

set -e
cd /app

# Resolve model path. Download to /app/models/ if not already present.
MODEL_DIR="/app/models"
MODEL_FILE="$MODEL_DIR/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Llama 3.2 1B Q4_K_M model not found at $MODEL_FILE, downloading (~700MB)..."
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$MODEL_FILE.tmp" \
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    else
        wget -O "$MODEL_FILE.tmp" \
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    fi
    mv "$MODEL_FILE.tmp" "$MODEL_FILE"
    echo "Model downloaded: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "Model present: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
fi

# Pre-warm Julia precompilation cache (no-op if already compiled).
julia --project=. -e 'using Pkg; Pkg.precompile()' || true

# Start the server. LlamaInference.load_llama_model() runs at startup
# via start_llama_model() inside IngExuity.jl so the model is in memory
# before the HTTP listener accepts traffic.
exec julia --project=/app -e '
using IngExuity
IngExuity.start()
'
