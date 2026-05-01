# Context

## Current State
Phase 3 geometric importance score computation script (T3.1) is now implemented. The script (`scripts/compute_importance.py`) supports efficiently batching over large datasets to compute $\overline{\Delta}$, the IDF factor, and $\alpha_{idf}$. Plain BERT warmup is also fully set up.
Current default BERT warmup training config in `configs/train/bert_warmup.yaml` is tuned from recent Instacart probe behavior for faster but stable convergence:
- `lr=1.2e-3`, `batch_size=256`, `epochs=24`, `warmup_steps=1000`, `min_lr_ratio=0.02`
- `weight_decay=4e-3`, `dropout=0.22`, `label_smoothing=0.03`, `early_stop_patience=4`

Config hygiene update completed: W&B-exported run snapshot YAMLs are now archived under `results/run_configs/bert_warmup/` and removed from `configs/` to avoid mixing experiment artifacts with Hydra config groups.

## Completed Tasks

**Phase 0 — Repo Bootstrap:**
- T0.1: uv init, pyproject.toml, Makefile
- T0.2: nbr/utils/seed.py with seed_everything()
- T0.3: nbr/utils/device.py with get_device() -> returns cpu
- T0.4: configs/ with all YAML files (data, model, train, config.yaml)

**Phase 1 — Data:**
- T1.1: scripts/download_data.py with instructions for Instacart, Dunnhumby, TaFeng
- T1.2: scripts/preprocess.py — Instacart preprocessor
- T1.3: scripts/preprocess.py — Dunnhumby and TaFeng preprocessors (shared _preprocess_basket_df helper)
- T1.4: nbr/data/split.py — split_user_baskets() with last=test, second-to-last=val, rest=train
- T1.5: nbr/data/dataset.py — BasketSequenceDataset and BasketCollator

**Phase 2 — Baselines (Model Components):**
- T2.1: nbr/models/embeddings.py — ItemEmbedding + Word2VecTrainer
- T2.2: nbr/models/encoder.py — IntraBasketEncoder with CLS token, TransformerEncoderLayer
- T2.3: nbr/models/gpt.py — RoPEAttention + CausalBasketGPT (causality verified)
- T2.4: nbr/models/vanilla.py — VanillaNBR
- T2.6: bert_gpt vanilla model scaffold exists in `nbr/models/bert_gpt/model.py`

**Phase 2.5 — Pre-training:**
- T2.7: scripts/train_word2vec.py implemented
- T2.8: Word2Vec pre-training completed for Instacart/TaFeng/Dunnhumby (`word2vec_dim128.kv` present)

**Phase 3 — Plain BERT Warmup & Importance (new):**
- Added `nbr/data/basket_mlm_dataset.py`: basket-level MLM dataset/collator (each basket = sentence)
- Added `nbr/models/bert.py`: BasketBERT model (item embedding + IntraBasketEncoder + tied MLM head)
- Added `nbr/train/bert_data_module.py`: lightweight dataloader builder (plain PyTorch)
- Added `scripts/train_bert.py`: Hydra entrypoint with plain PyTorch train/val loops, W&B logging, checkpointing, sanity checks, encoder bundle export
- Added configs: `configs/model/bert.yaml`, `configs/train/bert_warmup.yaml`, `configs/bert_warmup.yaml`
- Removed `nbr/train/bert_lightning.py` (no-Lightning policy)
- Added `results/bert_warmup.md` with smoke-run outputs and metrics
- T3.1: `scripts/compute_importance.py` implemented to compute `alpha_idf` scores using the pre-trained `IntraBasketEncoder` and batched masking strategy.

## Latest Run Snapshot
Dataset: Instacart probe (4 epochs)

Command:
`PYTHONPATH=. python scripts/train_bert.py data=instacart train.run_name=bert-warmup-instacart-bs256-lr12e4-e4-probe train.batch_size=128 train.num_workers=4 train.epochs=4 train.lr=1.2e-3 train.warmup_steps=200 train.min_lr_ratio=0.05 train.weight_decay=4.0e-3 train.dropout=0.22 train.label_smoothing=0.04 train.mask_prob=0.15 train.val_mask_prob=0.15 train.early_stop_patience=0 train.log_every_n_steps=20`

Observed epoch logs (key trend):
- epoch1: train_loss 7.4764, val_loss 7.1151, train_acc 0.1102, val_acc 0.1171
- epoch2: train_loss 7.2315, val_loss 6.9810, train_acc 0.1191, val_acc 0.1206
- epoch3: train_loss 7.1355, val_loss 6.9196, train_acc 0.1217, val_acc 0.1229
- epoch4: train_loss 7.0734, val_loss 6.8816, train_acc 0.1235, val_acc 0.1230

Diagnosis from probe:
- stable optimization with no overfitting signal in first 4 epochs
- tiny train/val gap, so previous stronger regularization looked somewhat underfitting
- very long epoch wall-clock on large datasets is expected; not a model stall

Recent full-run snapshots archived:
- `results/run_configs/bert_warmup/tafeng_2026-04-20_00-12-59.yaml`
- `results/run_configs/bert_warmup/dunnhumby_2026-04-20_00-56-51.yaml`

W&B run URLs and local output dirs are still discoverable in those snapshot files.

Artifacts from full runs remain under `outputs/<date>/<time>/` with:
- `bert_best.pt`
- `bert_last.pt`
- `bert_encoder_bundle_<dataset>.pt`
- `bert_sanity_checks.txt`

For downstream collaborator convenience, `bert_best.pt` and `bert_encoder_bundle_<dataset>.pt`
are also being placed under `data/processed/<dataset>/` on the training machine after runs.

## Next Task
T3.2 — Importance head module
Create `nbr/models/importance.py` containing:
- `ImportanceHead(nn.Module)`: A two-layer MLP (`Linear(D, D//2)` → `GELU` → `Linear(D//2, 1)` → `Sigmoid`) mapping item representations `(B*T, S, D)` to importance weights `(B*T, S)` in `[0, 1]`.
- `importance_init_loss`: An MSE loss function to pre-train the `ImportanceHead` by pushing the predicted weights towards the `alpha_idf` target scores (computed in T3.1), masked to real items only.
## Decisions Made
- Current scope restricted to plain BERT warmup baseline only.
- BERT uses basket-as-sentence setup: item tokens per basket, CLS output as basket representation.
- Word2Vec initialization is required for item embeddings when `.kv` exists.
- Training and tracking requirements: Hydra + plain PyTorch loops + W&B.
- New decision: do not use PyTorch Lightning for new training loops in this repo.
- Keep code simple, modular, and notebook-portable for possible Kaggle execution.

## Broken / Incomplete
- Old Lightning-based scripts still exist (`scripts/train_vanilla.py`, `scripts/train_importance.py`) and are not yet migrated.
- BERT warmup full runs for all datasets pending.
- Unit tests for new BERT warmup modules not yet added.

## Best Val Metrics
- Plain BERT warmup (TaFeng smoke subset): val_mlm_loss 9.53411
- Recent Instacart 4-epoch probe improved steadily to val_mlm_loss 6.8816 at epoch 4 (probe only, not final full run metric).

## Baseline Metrics (frequency)
- Full-dataset frequency baseline metrics and GPTopFreq alpha sweeps logged in `results/baseline_eval.md`
