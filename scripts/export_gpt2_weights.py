"""Export GPT-2 pretrained weights to Julia-compatible binary format."""
import json
import os
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

MODEL_NAME = "gpt2"
OUTPUT_FILE = "models/gpt2_124m.bin"
TOKENIZER_DIR = "models/gpt2_tokenizer"


def export_weights():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = GPT2Model.from_pretrained(MODEL_NAME, output_attentions=False)
    model.eval()

    state_dict = model.state_dict()
    tensor_map = {}

    for name, tensor in state_dict.items():
        arr = tensor.detach().numpy().astype(np.float32)
        tensor_map[name] = arr
        print(f"  {name}: {arr.shape}")

    header = {}
    all_data = b""
    for name in sorted(tensor_map.keys()):
        arr = tensor_map[name]
        header[name] = {"shape": list(arr.shape)}
        all_data += arr.tobytes()

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(header_bytes)
        f.write(b"\x00")
        f.write(all_data)

    size_mb = (len(header_bytes) + 1 + len(all_data)) / (1024 * 1024)
    print(f"\nExported {len(tensor_map)} tensors to {OUTPUT_FILE} ({size_mb:.1f} MB)")


def export_tokenizer():
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    vocab = tokenizer.get_vocab()
    with open(os.path.join(TOKENIZER_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  vocab.json: {len(vocab)} entries")

    tokenizer.save_pretrained(TOKENIZER_DIR)
    tok_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    merges_raw = tok_data["model"]["merges"]
    with open(os.path.join(TOKENIZER_DIR, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges_raw:
            if isinstance(m, str):
                f.write(m + "\n")
            else:
                f.write(f"{m[0]} {m[1]}\n")
    print(f"  merges.txt: {len(merges_raw)} merges")

    print(f"Tokenizer saved to {TOKENIZER_DIR}/")


if __name__ == "__main__":
    export_weights()
    export_tokenizer()
