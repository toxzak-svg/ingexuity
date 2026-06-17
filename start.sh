#!/bin/bash
set -e

cd /app

# Load pretrained GPT-2 weights if available
if [ -f models/gpt2_124m.bin ]; then
    echo "GPT-2 weights found."
fi

# Start Julia server (loads NanoGPT pretrained weights automatically)
exec julia --project=. -e '
using IngExuity

# Load pretrained GPT-2 weights if available
if isfile("models/gpt2_124m.bin")
    @info "Loading GPT-2 pretrained model..."
    IngExuity.load_local_model("models/gpt2_124m.bin")
end

IngExuity.start()
'