# Kaggle Synthetic Prediction Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, train, and package a small synthetic IngExuity prediction dataset on Kaggle before scaling to the approved 10,000-example pilot.

**Architecture:** A deterministic scenario engine owns ground-truth labels; a renderer converts scenarios into natural multi-turn records; validators reject malformed or unsafe rows; and a family-aware splitter prevents scenario leakage. Kaggle first runs a 100-example smoke artifact, then trains the existing Llama 3.2 1B QLoRA notebook only if validation passes. Runtime prediction-envelope integration and the audit-log viewer are follow-on plans.

**Tech Stack:** Python 3.10+, standard library, pytest, Hugging Face Transformers/Datasets/PEFT, Kaggle notebooks, JSON/JSONL

---

## File Structure

- `python/ingexuity_data/__init__.py`: public dataset-pipeline exports.
- `python/ingexuity_data/schema.py`: typed record constructors and probability validation.
- `python/ingexuity_data/scenarios.py`: deterministic scenario families and labeled outcomes.
- `python/ingexuity_data/render.py`: template renderer and provider-neutral teacher interface.
- `python/ingexuity_data/validate.py`: structural, safety, leakage, and duplicate checks.
- `python/ingexuity_data/build.py`: dataset assembly, family-aware splitting, hashes, and manifest.
- `python/build_synthetic_dataset.py`: command-line entry point used locally and on Kaggle.
- `python/test_ingexuity_data.py`: focused unit and pipeline tests.
- `models/trained_model/notebooks/train_weights.ipynb`: add a synthetic-data source option and preserve existing upload/path/GitHub modes.
- `docs/training/SYNTHETIC_DATA.md`: operator instructions and claim boundaries.

### Task 1: Define the Prediction Record Contract

**Files:**
- Create: `python/ingexuity_data/__init__.py`
- Create: `python/ingexuity_data/schema.py`
- Create: `python/test_ingexuity_data.py`

- [ ] **Step 1: Write failing schema tests**

```python
from ingexuity_data.schema import normalize_probabilities, validate_record


def test_normalize_probabilities_sums_to_one():
    values = normalize_probabilities([4, 3, 3])
    assert values == [0.4, 0.3, 0.3]


def test_rejects_unconfirmed_real_world_outcome():
    record = valid_record()
    record["observed_outcome"]["real_world"] = "started project"
    record["observed_outcome"]["real_world_status"] = "unknown"
    errors = validate_record(record)
    assert "unknown real-world outcome must not contain an observed action" in errors
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: collection fails because `ingexuity_data.schema` does not exist.

- [ ] **Step 3: Implement schema validation**

```python
def normalize_probabilities(weights: list[float]) -> list[float]:
    if not weights or any(value < 0 for value in weights):
        raise ValueError("probability weights must be non-empty and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    normalized = [value / total for value in weights]
    normalized[-1] += 1.0 - sum(normalized)
    return normalized


def validate_record(record: dict) -> list[str]:
    errors = []
    required = {
        "scenario_id", "family", "conversation", "known_user_facts",
        "inferred_user_state", "predictions", "response_mode",
        "planned_action", "assistant_response", "observed_outcome",
        "user_model_update",
    }
    missing = sorted(required - record.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    for group in ("conversation", "real_world"):
        candidates = record.get("predictions", {}).get(group, [])
        total = sum(item.get("probability", -1) for item in candidates)
        if len(candidates) < 2 or abs(total - 1.0) > 1e-6:
            errors.append(f"{group} predictions require alternatives summing to one")
    outcome = record.get("observed_outcome", {})
    if outcome.get("real_world_status") == "unknown" and outcome.get("real_world") not in (None, "unknown"):
        errors.append("unknown real-world outcome must not contain an observed action")
    return errors
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: schema tests pass.

- [ ] **Step 5: Commit the schema slice**

```bash
git add python/ingexuity_data/__init__.py python/ingexuity_data/schema.py python/test_ingexuity_data.py
git commit -m "feat: define synthetic prediction record schema"
```

### Task 2: Generate Deterministic Scenario Families

**Files:**
- Create: `python/ingexuity_data/scenarios.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing distribution and determinism tests**

```python
from ingexuity_data.scenarios import generate_scenarios


def test_scenarios_are_reproducible_and_balanced():
    first = generate_scenarios(count=100, seed=42)
    second = generate_scenarios(count=100, seed=42)
    assert first == second
    assert len({item["scenario_id"] for item in first}) == 100
    modes = {item["response_mode"] for item in first}
    assert modes == {"presence", "action", "balanced"}
    assert any(item["real_world_status"] == "unknown" for item in first)
    assert any(item["scenario_kind"] == "recovery" for item in first)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest python/test_ingexuity_data.py::test_scenarios_are_reproducible_and_balanced -v`

Expected: FAIL because `scenarios.py` does not exist.

- [ ] **Step 3: Implement seeded scenario generation**

Define family templates for work avoidance, emotional disclosure, planning, relationship tension, health uncertainty, routine disruption, preference change, correction, and explicit mode overrides. Cycle families and modes before sampling wording so a 100-row smoke includes every family and mode. Use `random.Random(seed)` exclusively and derive IDs as `family-kind-index-seed`.

```python
SCENARIO_FAMILIES = {
    "work_avoidance": {
        "evidence": "I keep opening the project and then closing it.",
        "state": {"emotion": "overwhelmed", "need": "reduce activation energy"},
        "conversation_outcomes": ["describes blocker", "asks where to begin", "withdraws"],
        "real_world_outcomes": ["attempts small task", "continues avoiding", "seeks context"],
    },
    "emotional_disclosure": {"evidence": "I don't need advice. I just need to say this somewhere."},
    "planning": {"evidence": "I have three deadlines and need to choose what comes first."},
    "relationship_tension": {"evidence": "They said they're fine, but the conversation felt wrong."},
    "health_uncertainty": {"evidence": "I've felt off today and I'm not sure whether to worry."},
    "routine_disruption": {"evidence": "My normal routine disappeared this week."},
    "preference_change": {"evidence": "I know I used to want reminders, but they're irritating me now."},
    "correction": {"evidence": "No, that's not why I'm avoiding it."},
    "explicit_override": {"evidence": "Don't solve this yet. Just stay with me."},
}
```

Each compact entry is expanded by `generate_scenarios` into the same required keys shown for `work_avoidance`: state, three conversational outcomes, three real-world outcomes, mode, outcome status, and scenario kind. The test asserts those expanded keys for every emitted scenario.

- [ ] **Step 4: Run all focused tests**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: all schema and scenario tests pass.

- [ ] **Step 5: Commit the scenario engine**

```bash
git add python/ingexuity_data/scenarios.py python/test_ingexuity_data.py
git commit -m "feat: generate labeled prediction scenarios"
```

### Task 3: Render Natural Records Without Losing Labels

**Files:**
- Create: `python/ingexuity_data/render.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing renderer tests**

```python
from ingexuity_data.render import TemplateRenderer


def test_renderer_keeps_internal_fields_out_of_visible_response():
    scenario = generate_scenarios(1, seed=7)[0]
    record = TemplateRenderer(seed=7).render(scenario)
    assert record["assistant_response"]
    assert "confidence" not in record["assistant_response"].lower()
    assert "predictions" not in record["assistant_response"].lower()
    assert not validate_record(record)
```

- [ ] **Step 2: Verify renderer test fails**

Run: `python -m pytest python/test_ingexuity_data.py::test_renderer_keeps_internal_fields_out_of_visible_response -v`

Expected: FAIL because `render.py` does not exist.

- [ ] **Step 3: Implement renderer boundary**

```python
class Renderer(Protocol):
    def render(self, scenario: dict) -> dict: ...


class TemplateRenderer:
    def __init__(self, seed: int):
        self.random = random.Random(seed)

    def render(self, scenario: dict) -> dict:
        # Select phrasing from mode-specific pools while copying scenario-owned labels.
        # The response contains natural language only; internal labels remain separate.
        return build_record_from_scenario(scenario, self.random)
```

Also define a `TeacherRenderer` interface accepting a callable `generate(prompt) -> str`. Require returned JSON to pass the same validator; do not add provider credentials or a provider-specific SDK to the core package.

- [ ] **Step 4: Run renderer and schema tests**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the renderer**

```bash
git add python/ingexuity_data/render.py python/test_ingexuity_data.py
git commit -m "feat: render natural synthetic prediction records"
```

### Task 4: Validate, De-duplicate, Split, and Manifest the Dataset

**Files:**
- Create: `python/ingexuity_data/validate.py`
- Create: `python/ingexuity_data/build.py`
- Create: `python/build_synthetic_dataset.py`
- Modify: `python/test_ingexuity_data.py`

- [ ] **Step 1: Add failing pipeline tests**

```python
def test_build_writes_family_isolated_splits_and_manifest(tmp_path):
    result = build_dataset(output_dir=tmp_path, count=100, seed=42)
    train_families = set(result["split_families"]["train"])
    eval_families = set(result["split_families"]["eval"])
    test_families = set(result["split_families"]["test"])
    assert train_families.isdisjoint(eval_families | test_families)
    assert eval_families.isdisjoint(test_families)
    assert result["accepted"] == 100
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "train.jsonl").is_file()
```

- [ ] **Step 2: Verify pipeline test fails**

Run: `python -m pytest python/test_ingexuity_data.py::test_build_writes_family_isolated_splits_and_manifest -v`

Expected: FAIL because `build.py` does not exist.

- [ ] **Step 3: Implement deterministic build and quarantine flow**

The build function generates scenarios, renders rows, validates each row, rejects exact and normalized near-duplicates, partitions whole families into train/eval/test splits, writes JSONL files, and records SHA-256 hashes. Invalid rows go to `quarantine.jsonl` with rejection reasons.

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

CLI:

```python
parser.add_argument("--output", default="synthetic_pilot")
parser.add_argument("--count", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--renderer", choices=["template"], default="template")
```

- [ ] **Step 4: Run the full Python test file**

Run: `python -m pytest python/test_ingexuity_data.py -v`

Expected: all tests pass.

- [ ] **Step 5: Build a local 100-row smoke dataset**

Run: `python python/build_synthetic_dataset.py --output .tmp/synthetic-pilot --count 100 --seed 42`

Expected: exit 0; manifest reports 100 accepted rows, zero schema failures, disjoint split families, and hashes for every split.

- [ ] **Step 6: Commit the pipeline**

```bash
git add python/ingexuity_data/validate.py python/ingexuity_data/build.py python/build_synthetic_dataset.py python/test_ingexuity_data.py
git commit -m "feat: build validated synthetic prediction datasets"
```

### Task 5: Integrate the Synthetic Source Into the Training Notebook

**Files:**
- Modify: `models/trained_model/notebooks/train_weights.ipynb`
- Create: `python/test_training_notebook.py`

- [ ] **Step 1: Add a failing notebook contract test**

```python
def test_notebook_supports_synthetic_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'DATA_SOURCE = "synthetic"' in source
    assert "build_synthetic_dataset.py" in source
    assert "synthetic_pilot/train.jsonl" in source
```

- [ ] **Step 2: Verify the contract test fails**

Run: `python -m pytest python/test_training_notebook.py -v`

Expected: FAIL because the synthetic source is absent.

- [ ] **Step 3: Add a synthetic source cell**

Add `synthetic` to the existing upload/path/GitHub data-source choices. The cell runs:

```python
!python python/build_synthetic_dataset.py \
    --output /kaggle/working/synthetic_pilot \
    --count 100 \
    --seed 42
config.data_path = "/kaggle/working/synthetic_pilot/train.jsonl"
```

The notebook must print the manifest and stop before model loading if `accepted != requested`, validation errors exist, or split-family overlap is non-empty.

- [ ] **Step 4: Run notebook contract tests**

Run: `python -m pytest python/test_training_notebook.py -v`

Expected: all notebook contract tests pass.

- [ ] **Step 5: Commit notebook integration**

```bash
git add models/trained_model/notebooks/train_weights.ipynb python/test_training_notebook.py
git commit -m "feat: add synthetic pilot to training notebook"
```

### Task 6: Document the Pilot and Its Claim Boundary

**Files:**
- Create: `docs/training/SYNTHETIC_DATA.md`
- Modify: `python/README.md`

- [ ] **Step 1: Document exact local and Kaggle commands**

Include:

```bash
python -m pytest python/test_ingexuity_data.py python/test_training_notebook.py -v
python python/build_synthetic_dataset.py --output synthetic_pilot --count 100 --seed 42
```

Document that the 100-row run proves pipeline execution only. It is not the approved 10,000-example pilot, not a quality result, and not evidence of improved user prediction.

- [ ] **Step 2: Verify documentation references resolve**

Run: `Select-String -Path docs/training/SYNTHETIC_DATA.md,python/README.md -Pattern 'build_synthetic_dataset.py|train_weights.ipynb|10,000'`

Expected: both documents reference the builder, notebook, and scale boundary.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/training/SYNTHETIC_DATA.md python/README.md
git commit -m "docs: explain synthetic prediction pilot workflow"
```

### Task 7: Run the Kaggle Smoke and Preserve Evidence

**Files:**
- Kaggle output: `/kaggle/working/synthetic_pilot/manifest.json`
- Kaggle output: `/kaggle/working/synthetic_pilot/*.jsonl`
- Kaggle output: `/kaggle/working/ingexuity_runs/<run-id>/my_weights/`
- Kaggle output: `/kaggle/working/ingexuity_runs/<run-id>/training_manifest.json`

- [ ] **Step 1: Open the existing Kaggle notebook and enable an NVIDIA GPU**

Expected: `torch.cuda.is_available()` prints `True`, and the notebook records the GPU model.

- [ ] **Step 2: Ensure Kaggle secrets and model access are available**

Expected: `HF_TOKEN` resolves without printing the token, and `meta-llama/Llama-3.2-1B-Instruct` tokenizer/model access succeeds.

- [ ] **Step 3: Run dataset generation and inspect validation summary**

Expected: 100 requested and accepted records, no schema errors, no family overlap, and non-empty train/eval/test files.

- [ ] **Step 4: Run one-epoch QLoRA smoke training**

Set `num_train_epochs = 1`, preserve the notebook's 4-bit QLoRA settings, and run through evaluation, smoke inference, and packaging.

Expected: training exits normally; evaluation loss is finite; adapter and tokenizer files exist; smoke inference returns non-empty text.

- [ ] **Step 5: Save the Kaggle notebook version and artifacts**

Expected: Kaggle output contains the synthetic manifest, dataset splits, training manifest, metrics, and adapter ZIP. Record the notebook version URL and exact run status in the handoff.

- [ ] **Step 6: Report only what the smoke proves**

The completion report must distinguish:

- pipeline artifact: generated, validated, trained, and packaged;
- model-quality result: not established by a 100-row synthetic smoke;
- next gate: establish frozen base-model metrics and run the approved 10,000-example pilot.

## Self-Review Result

- **Spec coverage:** This plan covers scenario generation, schema enforcement, family-aware splitting, provenance hashes, Colab/Kaggle-compatible training, and the first artifact smoke. Provider-specific teacher generation, the full 10,000-example run, runtime envelope parsing, outcome scoring, and the log viewer are intentionally deferred into separate plans.
- **Placeholder scan:** No implementation step relies on an unspecified TODO or TBD.
- **Type consistency:** Record fields use the names defined in the design specification; `real_world_status` explicitly controls whether a real-world outcome may contain an observed value.
