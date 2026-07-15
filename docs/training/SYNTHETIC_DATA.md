# Synthetic Prediction Training Data

IngExuity's synthetic-data pipeline creates labeled conversations for three separate capabilities:

- infer a user state while keeping hypotheses separate from facts;
- predict the next conversational move and possible real-world actions; and
- select Presence, Action, or Balanced behavior before producing the user-visible response.

The internal target is a JSON prediction envelope. The visible response stays in `assistant_response`; predictions and User Model updates remain internal.

## Run the focused checks

From the repository root:

```bash
python -m pytest python/test_ingexuity_data.py python/test_prediction_evaluation.py python/test_training_notebook.py -v
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

The template renderer is deliberately limited to pipeline smoke tests; repeating templates at scale would create low-value data.

## Build the teacher-rendered 10K pilot

On a GPU runtime, run:

```bash
python python/build_synthetic_dataset.py \
  --output /kaggle/working/synthetic_pilot \
  --count 10000 \
  --seed 42 \
  --renderer teacher \
  --teacher-model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher-batch-size 8 \
  --benchmark-count 200 \
  --require-scale-gate
```

The teacher may rewrite only `user_message` and `assistant_response`; deterministic code owns all labels. Accepted and quarantined attempts are appended immediately to `teacher-accepted.jsonl` and `teacher-quarantine.jsonl`, so rerunning the same command resumes completed scenario IDs.

The 200-row benchmark must satisfy all three frozen gates before the 10K phase starts:

- at least 95% accepted;
- no more than 2% duplicates; and
- projected 10K teacher runtime no more than eight hours.

The command exits nonzero if the benchmark fails or if fewer than 10,000 unique validated rows survive.

## Run on Kaggle

Open [`models/trained_model/notebooks/train_weights.ipynb`](../../models/trained_model/notebooks/train_weights.ipynb) in Kaggle and enable an NVIDIA GPU and Internet. The target defaults to the public `unsloth/Llama-3.2-1B-Instruct` mirror and the teacher to public `Qwen/Qwen2.5-1.5B-Instruct`, so API-launched Kaggle versions do not depend on account secrets. If `HF_TOKEN` is available, the notebook authenticates normally. The training manifest records the exact target source identifier.

The notebook defaults to `DATA_SOURCE = "synthetic"`. It clones the configured repository ref, applies the 200-row scale gate, builds the exact 10K curriculum, rejects invalid manifests or family overlap, uses held-out family splits, and trains for one epoch. After saving the adapter, it evaluates both the untouched base and adapter on the same frozen test records. Output is written beneath:

```text
/kaggle/working/synthetic_pilot/
/kaggle/working/ingexuity_training/
```

The adapter ZIP includes the dataset manifest, teacher scale gate, and base/adapter structured metrics. The frozen adapter gates are at least 80% valid envelopes, at least 75% mode accuracy, at most 1% internal leakage, and zero flagged manipulation. The adapter ZIP remains an experimental artifact even when packaging succeeds; do not replace the checked-in `my_weights` adapter unless the quality comparison passes review.

## Claim boundary

A completed 10K job proves that gated teacher generation, validation, family-aware splitting, QLoRA training, frozen evaluation, and packaging execute together. It proves better structured prediction behavior only if the adapter passes the fixed gates and improves on the untouched base metrics. Training loss or a saved ZIP alone is not a model-quality pass.
