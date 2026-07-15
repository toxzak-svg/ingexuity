# Kaggle Synthetic Prediction Smoke — 2026-07-15

## Run

- Kaggle notebook: `zacharymaronek/ingexuity-synthetic-prediction-pilot`, version 5
- Notebook URL: <https://www.kaggle.com/code/zacharymaronek/ingexuity-synthetic-prediction-pilot>
- Feature ref: `codex/kaggle-synthetic-pilot` at `046e4f8`
- Base source: `unsloth/Llama-3.2-1B-Instruct`
- GPU: Tesla P100-PCIE-16GB, 15.89 GiB
- Training: one epoch, effective batch size 32

## Dataset artifact

- Requested: 100
- Accepted: 100
- Rejected: 0
- Train/evaluation/test: 56/22/22
- Response modes: Action 34, Balanced 33, Presence 33
- Scenario kinds: Ambiguous 16, Contradiction 17, Override 17, Recovery 17, Safety 17, Success 16
- Family overlap: none
- Train SHA-256: `71cba0811e658739aac7d6f5433da5e40e77aba7363059cf50ff4ad2a2879b65`

## Training artifact

- Trainable parameters: 22,544,384 of 1,258,358,784
- Initial held-out loss: 3.987698554992676
- Initial held-out perplexity: 53.930628053213155
- Final held-out loss: 3.267582654953003
- Final held-out perplexity: 26.247812571423413
- Reported train loss: 3.930680513381958
- Training runtime: 33.9051 seconds
- Adapter: 90,207,248 bytes
- Packaged ZIP: 63,925,223 bytes

## Verdict

**Pipeline artifact: PASS.** Synthetic generation, deterministic validation, family-aware splitting, QLoRA training, held-out evaluation, adapter save, and packaging completed on Kaggle.

**Structured prediction behavior: FAIL.** The generation smoke produced a natural-language empathetic response rather than the required JSON prediction envelope. The run therefore does not establish that the adapter learned internal User Model updates, conversational predictions, real-world predictions, or mode routing.

**Model-quality claim: NOT ESTABLISHED.** The lower held-out loss is encouraging but comes from a 56-example template-rendered training split and 22-example held-out-family split. It is not sufficient evidence of general user-prediction quality.

## Next gate

1. Make the generation smoke request and validate the exact runtime prediction envelope.
2. Freeze base-model structured-validity, routing, prediction, calibration, leakage, and safety metrics.
3. Implement teacher-rendered natural variation.
4. Run the approved 10,000-example pilot.
5. Compare against the frozen baseline without weakening gates after seeing results.
