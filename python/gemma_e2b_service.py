#!/usr/bin/env python3
# ============================================================================
# gemma_e2b_service.py — Gemma 4 E2B HTTP service for IngExuity
# Runs Gemma 4 E2B with native function calling and audio support
# ============================================================================

import os
import sys
import json
import argparse
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import wave
import struct

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_PORT = 8765

# ---------------------------------------------------------------------------
# Gemma Model Loader
# ---------------------------------------------------------------------------

_model = None
_processor = None

def load_model():
    global _model, _processor
    if _model is not None:
        return

    print(f"[GemmaE2B] Loading {MODEL_ID}...", file=sys.stderr)
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype="float32",
            trust_remote_code=True
        )
        print(f"[GemmaE2B] Model loaded successfully", file=sys.stderr)
    except ImportError as e:
        print(f"[GemmaE2B] Missing dependencies: {e}", file=sys.stderr)
        print("[GemmaE2B] Run: pip install transformers torch accelerate", file=sys.stderr)
        raise

def unload_model():
    global _model, _processor
    _model = None
    _processor = None
    import gc
    gc.collect()

# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def save_audio_from_base64(audio_b64: str, path: str = "/tmp/ingesuity_input.wav") -> str:
    """Decode base64 audio and save as WAV"""
    import base64
    audio_bytes = base64.b64decode(audio_b64)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path

def generate_speech(text: str, output_path: str = "/tmp/ingesuity_output.wav") -> str:
    """Generate speech from text using Coqui/TTS or fall back to system"""
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/ljspeech", progress_bar=False)
        tts.tts_to_file(text=text, file_path=output_path)
        return output_path
    except ImportError:
        pass
    except Exception as e:
        print(f"[GemmaE2B] TTS warning: {e}", file=sys.stderr)

    print(f"[GemmaE2B] No TTS available, returning text only", file=sys.stderr)
    return ""

def audio_to_base64(wav_path: str) -> str:
    """Encode WAV file as base64"""
    import base64
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def generate_response(messages: list, max_new_tokens: int = 512, enable_thinking: bool = False) -> dict:
    """Generate response from Gemma 4 E2B"""
    if _model is None or _processor is None:
        return {"error": "Model not loaded", "text": "", "actions": []}

    try:
        # Build prompt from messages
        text = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        # Process input
        inputs = _processor(text=text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with np.errstate(over='ignore'):
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True
            )

        # Decode
        response = _processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse for actions (function calls)
        actions = parse_actions(response)

        # Extract just the response text (not special tokens)
        clean_text = _processor.parse_response(response)

        return {
            "text": clean_text,
            "raw": response,
            "actions": actions,
            "thinking": extract_thinking(response) if enable_thinking else None
        }

    except Exception as e:
        print(f"[GemmaE2B] Generation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": str(e), "text": "", "actions": []}

def parse_actions(response: str) -> list:
    """Parse <invoke=...> tags from response"""
    import re
    actions = []
    # Match <invoke name="function_name"><parameter name="x">value</parameter></invoke>
    invoke_pattern = r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>'
    for match in re.finditer(invoke_pattern, response, re.DOTALL):
        func_name = match.group(1)
        params_match = re.findall(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', match.group(2))
        params = {k: v for k, v in params_match}
        actions.append({
            "function": func_name,
            "parameters": params
        })
    return actions

def extract_thinking(response: str) -> str:
    """Extract thinking block from response"""
    import re
    match = re.search(r'<\|channel\|>thought\n(.*?)<\|channel\|>', response, re.DOTALL)
    return match.group(1) if match else ""

def generate_speech_action(text: str) -> dict:
    """Generate speech audio from text"""
    wav_path = generate_speech(text)
    if wav_path and os.path.exists(wav_path):
        audio_b64 = audio_to_base64(wav_path)
        return {
            "type": "audio",
            "audio": audio_b64,
            "format": "wav"
        }
    return {"type": "text", "text": text}

# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class GemmaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model": MODEL_ID})
        elif self.path == "/capabilities":
            self.send_json({
                "model": MODEL_ID,
                "features": [
                    "text_generation",
                    "function_calling",
                    "audio_input",
                    "speech_generation",
                    "thinking_mode"
                ]
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/generate":
            self.handle_generate(data)
        elif self.path == "/generate_audio":
            self.handle_generate_audio(data)
        elif self.path == "/load":
            self.handle_load()
        elif self.path == "/unload":
            self.handle_unload()
        else:
            self.send_json({"error": "Not found"}, 404)

    def handle_generate(self, data):
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        thinking = data.get("enable_thinking", False)

        result = generate_response(messages, max_tokens, thinking)
        self.send_json(result)

    def handle_generate_audio(self, data):
        """Handle audio input: transcribe + generate response + speech"""
        audio_b64 = data.get("audio", "")
        messages = data.get("messages", [])

        audio_path = ""
        if audio_b64:
            audio_path = save_audio_from_base64(audio_b64)

        # Build messages with audio if provided
        if audio_path:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Transcribe and respond to this."}
                ]
            })
        else:
            self.send_json({"error": "No audio provided"}, 400)
            return

        # Generate response
        result = generate_response(messages)

        # Generate speech for the response
        if result.get("text"):
            speech_result = generate_speech_action(result["text"])
            result["audio_response"] = speech_result

        self.send_json(result)

    def handle_load(self):
        try:
            load_model()
            self.send_json({"status": "loaded", "model": MODEL_ID})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_unload(self):
        unload_model()
        self.send_json({"status": "unloaded"})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B HTTP Service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--auto-load", action="store_true", help="Load model on startup")
    args = parser.parse_args()

    if args.auto_load:
        print("[GemmaE2B] Auto-loading model...", file=sys.stderr)
        load_model()

    server = HTTPServer(("0.0.0.0", args.port), GemmaHandler)
    print(f"[GemmaE2B] Server running on http://0.0.0.0:{args.port}", file=sys.stderr)
    print(f"[GemmaE2B] Model: {MODEL_ID}", file=sys.stderr)
    print(f"[GemmaE2B] Endpoints:", file=sys.stderr)
    print(f"[GemmaE2B]   GET  /health - health check", file=sys.stderr)
    print(f"[GemmaE2B]   GET  /capabilities - model capabilities", file=sys.stderr)
    print(f"[GemmaE2B]   POST /generate - text generation", file=sys.stderr)
    print(f"[GemmaE2B]   POST /generate_audio - audio input + speech output", file=sys.stderr)
    print(f"[GemmaE2B]   POST /load - load model", file=sys.stderr)
    print(f"[GemmaE2B]   POST /unload - unload model", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[GemmaE2B] Shutting down...", file=sys.stderr)
        server.shutdown()

if __name__ == "__main__":
    main()