# Teacher-Rendered 10K Prediction Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, validate, train, and evaluate the approved 10,000-example hybrid IngExuity prediction curriculum on Kaggle.

**Architecture:** Ground-truth state, prediction, outcome, and routing labels remain deterministic. A public local teacher model rewrites only observable dialogue and the visible response, so teacher errors cannot silently redefine labels. A 200-example Kaggle benchmark must pass quality, duplicate, and runtime gates before resumable 10K generation and one-epoch QLoRA training begin.

**Tech Stack:** Python 3.10+, pytest, Hugging Face Transformers, Qwen2.5-1.5B-Instruct teacher, Llama 3.2 1B QLoRA target, Kaggle P100/T4

---

## File Structure

- `python/ingexuity_data/curriculum.py`: exact category quotas and deterministic diversity dimensions.
- `python/ingexuity_data/teacher.py`: batched local teacher loading, constrained rewrite prompts, parsing, and retries.
- `python/ingexuity_data/render.py`: construct labeled records from accepted teacher rewrites.
- `python/ingexuity_data/build.py`: resumable shards, quarantine, scale gates, splits, and manifest provenance.
- `python/build_synthetic_dataset.py`: large-run CLI options.
- `python/ingexuity_data/evaluate.py`: structured-envelope validity and routing metrics.
- `python/evaluate_prediction_model.py`: base/adapter evaluation CLI.
- `python/test_ingexuity_data.py`: curriculum, resume, rendering, and scale-gate tests.
- `python/test_prediction_evaluation.py`: exact structured-metric tests.
- `models/trained_model/notebooks/train_weights.ipynb`: benchmark, 10K generation, training, and structured smoke.
- `docs/training/SYNTHETIC_DATA.md`: 10K operator workflow and gates.

### Task 1: Define Exact 10K Curriculum Quotas

**Files:**
- Create: `python/ingexuity_data/curriculum.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing quota and diversity tests**

```python
from collections import Counter
from ingexuity_data.curriculum import CATEGORY_COUNTS, build_curriculum


def test_10k_curriculum_has_exact_approved_quotas():
    scenarios = build_curriculum(count=10_000, seed=42)
    assert Counter(row["category"] for row in scenarios) == CATEGORY_COUNTS
    assert len({row["scenario_id"] for row in scenarios}) == 10_000


def test_curriculum_has_observable_diversity_without_counter_language():
    scenarios = build_curriculum(count=10_000, seed=42)
    assert len({row["surface_seed"] for row in scenarios}) == 10_000
    assert len({row["domain"] for row in scenarios}) >= 12
    assert len({row["communication_style"] for row in scenarios}) >= 6
    assert all("noticed this" not in row["context_signal"].lower() for row in scenarios)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest python/test_ingexuity_data.py -k curriculum -v`

Expected: import failure because `curriculum.py` does not exist.

- [ ] **Step 3: Implement deterministic quota scheduling**

```python
CATEGORY_COUNTS = {
    "labeled_prediction": 4000,
    "natural_dialogue": 2000,
    "ambiguous": 1500,
    "recovery": 1000,
    "preference_change": 1000,
    "safety_boundary": 500,
}

DOMAINS = (
    "work", "learning", "creative_project", "household", "routine", "relationship",
    "friendship", "family", "health_uncertainty", "money_stress", "travel", "decision",
    "communication", "rest", "habit_change", "technology",
)
COMMUNICATION_STYLES = ("brief", "reflective", "indirect", "blunt", "anxious", "analytical", "humorous", "guarded")
TIME_HORIZONS = ("minutes", "today", "this_week", "this_month", "longer_term")
EVIDENCE_LEVELS = ("weak", "mixed", "moderate", "strong")

def build_curriculum(count: int, seed: int = 42) -> list[dict]:
    if count != sum(CATEGORY_COUNTS.values()):
        raise ValueError("the production curriculum requires exactly 10,000 examples")
    # Cycle the Cartesian dimensions with coprime strides, then deterministically
    # shuffle category assignments. Each row receives a UUID5 surface_seed derived
    # from seed/category/index; no artificial occurrence counts enter dialogue.
```

- [ ] **Step 4: Run curriculum tests**

Run: `python -m pytest python/test_ingexuity_data.py -k curriculum -v`

Expected: exact quotas and diversity tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/ingexuity_data/curriculum.py python/test_ingexuity_data.py
git commit -m "feat: define 10k prediction curriculum"
```

### Task 2: Add a Label-Preserving Local Teacher Renderer

**Files:**
- Create: `python/ingexuity_data/teacher.py`
- Modify: `python/ingexuity_data/render.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing teacher parser tests**

```python
from ingexuity_data.teacher import parse_teacher_rewrite


def test_teacher_parser_accepts_only_two_surface_fields():
    value = parse_teacher_rewrite('{"user_message":"I keep putting it off.","assistant_response":"What makes the first step feel heavy?"}')
    assert set(value) == {"user_message", "assistant_response"}


def test_teacher_parser_rejects_label_edits_and_internal_leaks():
    with pytest.raises(ValueError):
        parse_teacher_rewrite('{"user_message":"x","assistant_response":"x","response_mode":"action"}')
    with pytest.raises(ValueError):
        parse_teacher_rewrite('{"user_message":"x","assistant_response":"{\\"predictions\\":[]}"}')
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest python/test_ingexuity_data.py -k teacher -v`

Expected: import failure because `teacher.py` does not exist.

- [ ] **Step 3: Implement constrained prompt, parser, and batch generator**

```python
TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_PROMPT_VERSION = "surface-rewrite-v1"
ALLOWED_TEACHER_KEYS = {"user_message", "assistant_response"}

def parse_teacher_rewrite(text: str) -> dict[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher output has no JSON object")
    value = json.loads(text[start:end + 1])
    if set(value) != ALLOWED_TEACHER_KEYS:
        raise ValueError("teacher may rewrite surface fields only")
    for key, content in value.items():
        if not isinstance(content, str) or not content.strip() or len(content) > 800:
            raise ValueError(f"invalid teacher field: {key}")
    if any(marker in value["assistant_response"] for marker in INTERNAL_LEAK_MARKERS):
        raise ValueError("teacher leaked internal structure")
    return value

class LocalTeacher:
    def __init__(self, model_name=TEACHER_MODEL, batch_size=8, seed=42):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": 0})
        self.batch_size = batch_size
        self.seed = seed

    def rewrite_batch(self, scenarios: list[dict]) -> list[dict | None]:
        # Render strict two-field prompts, batch-tokenize, generate at most 160 new
        # tokens with do_sample=False, and return None for rejected outputs.
```

- [ ] **Step 4: Make `TemplateRenderer.render_with_surface` preserve all labels**

```python
def render_with_surface(self, scenario: dict, surface: dict[str, str]) -> dict:
    record = self.render(scenario)
    record["conversation"][-1]["content"] = surface["user_message"]
    record["assistant_response"] = surface["assistant_response"]
    record["inferred_user_state"]["evidence"] = [surface["user_message"]]
    record["messages"][1]["content"] = surface["user_message"]
    record["messages"][2]["content"] = json.dumps(_make_envelope(record), ensure_ascii=False, separators=(",", ":"))
    return record
```

- [ ] **Step 5: Run teacher tests and commit**

Run: `python -m pytest python/test_ingexuity_data.py -k "teacher or surface" -v`

Expected: all selected tests pass.

```bash
git add python/ingexuity_data/teacher.py python/ingexuity_data/render.py python/test_ingexuity_data.py
git commit -m "feat: add label-preserving local teacher renderer"
```

### Task 3: Build Resumable Teacher Shards and Scale Gates

**Files:**
- Modify: `python/ingexuity_data/build.py`
- Modify: `python/build_synthetic_dataset.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing resume and gate tests**

```python
def test_teacher_build_resumes_without_regenerating_completed_ids(tmp_path, fake_teacher):
    build_teacher_dataset(tmp_path, count=20, seed=42, teacher=fake_teacher, stop_after=10)
    first_calls = list(fake_teacher.calls)
    result = build_teacher_dataset(tmp_path, count=20, seed=42, teacher=fake_teacher)
    assert result["accepted"] == 20
    assert not set(first_calls) & set(fake_teacher.calls[len(first_calls):])


def test_benchmark_gate_rejects_low_acceptance_or_slow_eta():
    assert benchmark_gate(requested=200, accepted=189, duplicates=0, elapsed_seconds=20)["pass"] is False
    assert benchmark_gate(requested=200, accepted=195, duplicates=1, elapsed_seconds=700)["pass"] is False
    assert benchmark_gate(requested=200, accepted=196, duplicates=2, elapsed_seconds=300)["pass"] is True
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest python/test_ingexuity_data.py -k "resume or benchmark_gate" -v`

Expected: missing function failures.

- [ ] **Step 3: Implement JSONL checkpointing and gates**

```python
def benchmark_gate(requested, accepted, duplicates, elapsed_seconds):
    acceptance = accepted / requested
    duplicate_rate = duplicates / requested
    estimated_10k_hours = elapsed_seconds * (10_000 / requested) / 3600
    return {
        "acceptance_rate": acceptance,
        "duplicate_rate": duplicate_rate,
        "estimated_10k_hours": estimated_10k_hours,
        "pass": acceptance >= 0.95 and duplicate_rate <= 0.02 and estimated_10k_hours <= 8.0,
    }
```

Write accepted rows immediately to `teacher-accepted.jsonl`, rejected attempts to `teacher-quarantine.jsonl`, and completed scenario IDs to `teacher-state.json`. On restart, load these files and skip completed IDs. Production output is materialized only after all 10,000 requested IDs have either accepted rows or explicit rejection records; the CLI exits nonzero unless 10,000 valid unique rows survive.

- [ ] **Step 4: Extend CLI**

```python
parser.add_argument("--renderer", choices=["template", "teacher"], default="template")
parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--teacher-batch-size", type=int, default=8)
parser.add_argument("--benchmark-count", type=int, default=200)
parser.add_argument("--require-scale-gate", action="store_true")
```

- [ ] **Step 5: Run pipeline tests and commit**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: all dataset tests pass.

```bash
git add python/ingexuity_data/build.py python/build_synthetic_dataset.py python/test_ingexuity_data.py
git commit -m "feat: add resumable teacher dataset generation"
```

### Task 4: Freeze Structured Prediction Evaluation

**Files:**
- Create: `python/ingexuity_data/evaluate.py`
- Create: `python/evaluate_prediction_model.py`
- Create: `python/test_prediction_evaluation.py`

- [ ] **Step 1: Add failing exact metric tests**

```python
def test_evaluator_scores_valid_envelope_and_mode():
    result = score_generation(json.dumps(valid_envelope(mode="balanced")), expected_mode="balanced")
    assert result == {"structured_valid": 1, "mode_correct": 1, "internal_leak": 0}


def test_evaluator_rejects_natural_language_and_bad_probabilities():
    assert score_generation("I hear you.", "presence")["structured_valid"] == 0
    bad = valid_envelope(mode="presence")
    bad["predictions"]["conversation"][0]["probability"] = 2.0
    assert score_generation(json.dumps(bad), "presence")["structured_valid"] == 0
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest python/test_prediction_evaluation.py -v`

Expected: import failure because the evaluator does not exist.

- [ ] **Step 3: Implement frozen metrics**

`score_generation` parses exactly one object, requires the five runtime envelope keys, validates both prediction groups and probability sums, verifies the requested mode, and scans `assistant_response` for internal field names. The CLI runs the same held-out test JSONL against the untouched base and the adapter, writes JSON metrics, and never adjusts thresholds from observed results.

Required pilot gates:

```python
PILOT_GATES = {
    "structured_valid_rate": 0.80,
    "mode_accuracy": 0.75,
    "internal_leak_rate_max": 0.01,
    "manipulation_rate_max": 0.00,
}
```

- [ ] **Step 4: Run evaluator tests and commit**

Run: `python -m pytest python/test_prediction_evaluation.py -v`

Expected: all tests pass.

```bash
git add python/ingexuity_data/evaluate.py python/evaluate_prediction_model.py python/test_prediction_evaluation.py
git commit -m "feat: freeze structured prediction evaluation"
```

### Task 5: Add the Gated 10K Kaggle Path

**Files:**
- Modify: `models/trained_model/notebooks/train_weights.ipynb`
- Modify: `python/test_training_notebook.py`

- [ ] **Step 1: Add failing notebook contract tests**

```python
def test_notebook_has_10k_teacher_gate_and_structured_eval():
    source = notebook_source()
    assert "SYNTHETIC_PILOT_COUNT = 10_000" in source
    assert 'SYNTHETIC_RENDERER = "teacher"' in source
    assert "require-scale-gate" in source
    assert "evaluate_prediction_model.py" in source
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest python/test_training_notebook.py -k 10k -v`

Expected: assertion failure because the 10K path is absent.

- [ ] **Step 3: Add benchmark then production cells**

The notebook first generates 200 teacher rows and loads `benchmark_gate.json`. It raises before the full run unless `pass` is true. The production command is:

```python
subprocess.run([
    sys.executable, str(builder), "--output", str(output_dir),
    "--count", "10000", "--seed", str(config.seed),
    "--renderer", "teacher", "--teacher-batch-size", "8",
    "--benchmark-count", "200", "--require-scale-gate",
], cwd=str(source_root), check=True)
```

After adapter save, invoke `evaluate_prediction_model.py` on the held-out test file for both base and adapter. Package teacher provenance, dataset manifest, base metrics, adapter metrics, and the adapter into the output ZIP.

- [ ] **Step 4: Run all notebook contract tests and commit**

Run: `python -m pytest python/test_training_notebook.py -v`

Expected: all notebook code cells compile and all contracts pass.

```bash
git add models/trained_model/notebooks/train_weights.ipynb python/test_training_notebook.py
git commit -m "feat: add gated 10k Kaggle training path"
```

### Task 6: Run the 200-Example Kaggle Teacher Benchmark

- [ ] Push the branch and launch a private Kaggle version with GPU and Internet.
- [ ] Download `benchmark_gate.json`, teacher accepted/quarantine shards, and logs.
- [ ] Verify acceptance at least 95%, duplicates at most 2%, and estimated 10K runtime at most eight hours.
- [ ] If any gate fails, stop and record the failure; do not start 10K generation.
- [ ] If all gates pass, launch the resumable 10K version.

### Task 7: Run, Evaluate, and Report the 10K Pilot

- [ ] Monitor the Kaggle worker until COMPLETE or a concrete error.
- [ ] Download the dataset manifest, all split hashes, base metrics, adapter metrics, training manifest, adapter, and ZIP.
- [ ] Verify exact category quotas, 10,000 accepted unique rows, zero split-family overlap, and all required files.
- [ ] Record pipeline status and each model-quality gate separately in `docs/training/runs/2026-07-15-kaggle-teacher-10k.md`.
- [ ] Never replace the checked-in adapter automatically; keep the result experimental until the user reviews the evaluation.

## Self-Review Result

- **Spec coverage:** The plan covers the approved exact 10K mixture, label-preserving teacher variation, resumability, rejection/quarantine, provenance, split isolation, baseline comparison, structured validation, calibration-adjacent probability checks, leakage, and safety gates.
- **Scope boundary:** Runtime log-viewer implementation remains separate; this plan produces and evaluates the model artifact only.
- **Type consistency:** Teacher output contains only `user_message` and `assistant_response`; deterministic fields retain the existing record and runtime-envelope names.
