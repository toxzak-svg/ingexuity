# Synthetic Prediction Training Data

IngExuity's synthetic-data pipeline creates labeled conversations for three separate capabilities:

- infer a user state while keeping hypotheses separate from facts;
- predict the next conversational move and possible real-world actions; and
- select Presence, Action, or Balanced behavior before producing the user-visible response.

The internal target is a JSON prediction envelope. The visible response stays in `assistant_response`; predictions and User Model updates remain internal.

## Run the focused checks

From the repository root:

```bash
python -m pytest python/test_ingexuity_data.py python/test_training_notebook.py -v
```

## Build the local smoke dataset

```bash
python python/build_synthetic_dataset.py \
  --output data/synthetic-pilot \
  --count 100 \
  --seed 42
```

The command succeeds only when every requested row is accepted and the train, evaluation, and test families are disjoint. It writes:

- `train.jsonl`, `eval.jsonl`, and `test.jsonl`;
- `quarantine.jsonl` with rejection reasons;
- `manifest.json` with counts, scenario coverage, SHA-256 hashes, and the claim boundary.

The template renderer is deliberately limited to pipeline smoke tests. The approved 10,000-example pilot requires the provider-neutral teacher renderer plus frozen base-model evaluation; repeating templates at that scale would create low-value data.

## Run on Kaggle

Open [`models/trained_model/notebooks/train_weights.ipynb`](../../models/trained_model/notebooks/train_weights.ipynb) in Kaggle and enable an NVIDIA GPU and Internet. Add `HF_TOKEN` as a Kaggle secret after accepting access to `meta-llama/Llama-3.2-1B-Instruct`.

The notebook defaults to `DATA_SOURCE = "synthetic"`. It clones the configured repository ref, builds 100 records, rejects invalid manifests or family overlap, uses the held-out evaluation-family file, and forces one training epoch. Output is written beneath:

```text
/kaggle/working/synthetic_pilot/
/kaggle/working/ingexuity_training/
```

The adapter ZIP is an experimental artifact. Do not replace the checked-in `my_weights` adapter until its metrics and structured-output behavior have been reviewed.

## Claim boundary

A successful 100-row run proves that generation, validation, family-aware splitting, QLoRA training, evaluation, smoke inference, and packaging execute together. It does **not** prove that the adapter predicts users better than the base model.

The next quality gate is:

1. freeze base-model prediction, calibration, routing, leakage, and safety metrics;
2. implement teacher-rendered natural variation;
3. generate the approved 10,000-example pilot;
4. compare the adapter with the frozen baseline without weakening thresholds.
