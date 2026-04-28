# Gemma 4 E2B Service

HTTP service wrapping Google's Gemma 4 E2B model for IngExuity.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run service (will download model on first run)
python gemma_e2b_service.py --auto-load

# Or run without auto-loading, load via HTTP
python gemma_e2b_service.py
curl -X POST http://localhost:8765/load
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/capabilities` | Model capabilities |
| POST | `/generate` | Text generation |
| POST | `/generate_audio` | Audio input + speech output |
| POST | `/load` | Load model |
| POST | `/unload` | Unload model |

## Examples

### Text Generation

```bash
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are IngExuity."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256
  }'
```

### Audio Input

```bash
curl -X POST http://localhost:8765/generate_audio \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "<base64 audio>",
    "messages": [],
    "max_tokens": 256
  }'
```

### Function Calling

The model can call functions. Response includes:

```json
{
  "text": "Let me check that for you.",
  "actions": [
    {
      "function": "check_memory",
      "parameters": {"query": "user preferences"}
    }
  ]
}
```

## Running from Julia

```julia
using IngExuity.GemmaProvider

llm = GemmaLLM(port=8765)
load_model(llm)

response = generate(llm, "You are IngExuity.", "Hello!")
println(response["text"])
```