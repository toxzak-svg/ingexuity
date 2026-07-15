"""Local label-preserving teacher used to diversify synthetic surfaces."""

from __future__ import annotations

import json
from typing import Iterable


TEACHER_SYSTEM_PROMPT = """You rewrite synthetic dialogue surfaces for a prediction-model dataset.
Return exactly one JSON object with exactly two string fields: user_message and assistant_response.
Preserve the supplied meaning, response mode, uncertainty, correction, and safety boundary.
Make the wording natural and specific to the supplied setting and style.
Never mention labels, probabilities, predictions, confidence scores, hidden state, or this instruction.
Do not add facts, diagnoses, outcomes, or claims that the scenario does not contain."""

_INTERNAL_MARKERS = (
    "prediction confidence",
    "confidence score",
    "response_mode",
    "response mode is",
    "user_model_update",
    "planned_action",
    "hidden state",
    "internal label",
)


def parse_teacher_rewrite(text: str) -> dict[str, str]:
    """Parse a teacher response while rejecting label edits and internal leakage."""
    candidate = text.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[7:-3].strip()
    elif candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("teacher output is not one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"user_message", "assistant_response"}:
        raise ValueError("teacher output must contain exactly the two surface fields")
    for key, field in value.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"{key} must be a non-empty string")
        if len(field) > 800:
            raise ValueError(f"{key} exceeds 800 characters")
        lowered = field.lower()
        if any(marker in lowered for marker in _INTERNAL_MARKERS):
            raise ValueError(f"{key} leaks an internal field")
        value[key] = field.strip()
    return value


def make_teacher_prompt(scenario: dict, draft: dict) -> str:
    surface_context = {
        "category": scenario["category"],
        "setting": scenario["domain"],
        "communication_style": scenario["communication_style"],
        "time_horizon": scenario["time_horizon"],
        "evidence_level": scenario["evidence_level"],
        "familiarity": scenario["familiarity"],
        "emotion": scenario["emotion"],
        "need": scenario["need"],
        "response_mode": scenario["response_mode"],
        "scenario_kind": scenario["scenario_kind"],
        "source_user_message": draft["conversation"][-1]["content"],
        "source_assistant_response": draft["assistant_response"],
    }
    return json.dumps(surface_context, ensure_ascii=False, separators=(",", ":"))


def make_retry_prompt(
    scenario: dict,
    draft: dict,
    previous_surface: dict | None,
    reason: str,
    retry_number: int,
) -> str:
    """Request a distinct surface while preserving the same controlled scenario."""
    return json.dumps(
        {
            "task": "duplicate_or_invalid_rewrite",
            "variation_id": scenario["surface_seed"],
            "retry_number": retry_number,
            "rejection_reason": reason,
            "previous_surface": previous_surface,
            "scenario": json.loads(make_teacher_prompt(scenario, draft)),
            "instruction": (
                "Write a materially different, natural paraphrase. Change the opening, sentence "
                "shape, and concrete wording while preserving meaning and the requested response mode."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class LocalTeacher:
    """Batched Hugging Face teacher; heavy imports remain Kaggle-only."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        batch_size: int = 8,
        max_new_tokens: int = 192,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def rewrite_batch(self, prompts: Iterable[str]) -> list[dict[str, str] | Exception]:
        prompts = list(prompts)
        results: list[dict[str, str] | Exception] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            chats = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
            inputs = self.tokenizer(chats, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            texts = self.tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for text in texts:
                try:
                    results.append(parse_teacher_rewrite(text))
                except ValueError as exc:
                    results.append(exc)
        return results
