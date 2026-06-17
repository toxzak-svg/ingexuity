#!/bin/bash
# Download GPT-2 weights and export to Julia-compatible format
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if weights already exist
if [ -f "models/gpt2_124m.bin" ] && [ -f "models/gpt2_tokenizer/vocab.json" ]; then
    echo "GPT-2 weights already exist. Skipping download."
    exit 0
fi

# Install Python dependencies if needed
pip install -q transformers torch numpy 2>/dev/null || true

# Run export
echo "Downloading GPT-2 and exporting weights..."
python scripts/export_gpt2_weights.py

echo "Done. Weights saved to models/gpt2_124m.bin"
echo "Tokenizer saved to models/gpt2_tokenizer/"
