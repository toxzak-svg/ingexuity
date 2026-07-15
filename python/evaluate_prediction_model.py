#!/usr/bin/env python3
"""Evaluate a base or PEFT model on frozen held-out prediction records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingexuity_data.evaluate import contains_manipulation, score_generation, summarize_scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = [
        json.loads(line)
        for line in Path(args.test_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    scores = []
    manipulation_count = 0
    samples = []
    for row in rows:
        prompt_messages = row["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        score = score_generation(text, row["response_mode"])
        scores.append(score)
        manipulation_count += int(contains_manipulation(text))
        if len(samples) < 10:
            samples.append({"scenario_id": row["scenario_id"], "generation": text, "score": score})
    summary = summarize_scores(scores, manipulation_count)
    summary.update({"model": args.model, "adapter": args.adapter, "samples": samples})
    Path(args.output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
