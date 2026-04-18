# Context

## Current State
Phase 3 baseline work now focused on plain BERT warmup only. Word2Vec embeddings already exist for all datasets in `data/processed/{dataset}/word2vec_dim128.kv`. New plain PyTorch BERT warmup pipeline is implemented and smoke-tested on TaFeng with W&B logging. PyTorch Lightning is explicitly removed for this BERT path and disallowed for new training loops by project instruction. Latest updates fix post-training sanity check device mismatch on GPU runs, add clear per-epoch terminal timing/ETA prints, and introduce warmup+cosine LR with early stopping to reduce late-epoch validation degradation.

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

**Phase 3 — Plain BERT Warmup (new):**
- Added `nbr/data/basket_mlm_dataset.py`: basket-level MLM dataset/collator (each basket = sentence)
- Added `nbr/models/bert.py`: BasketBERT model (item embedding + IntraBasketEncoder + tied MLM head)
- Added `nbr/train/bert_data_module.py`: lightweight dataloader builder (plain PyTorch)
- Added `scripts/train_bert.py`: Hydra entrypoint with plain PyTorch train/val loops, W&B logging, checkpointing, sanity checks, encoder bundle export
- Added configs: `configs/model/bert.yaml`, `configs/train/bert_warmup.yaml`, `configs/bert_warmup.yaml`
- Removed `nbr/train/bert_lightning.py` (no-Lightning policy)
- Added `results/bert_warmup.md` with smoke-run outputs and metrics

## Latest Run Snapshot
Dataset: TaFeng smoke subset

Command:
`PYTHONPATH=. uv run python scripts/train_bert.py data=tafeng train.epochs=1 train.batch_size=64 train.num_workers=0 train.max_train_baskets=1500 train.max_val_baskets=300 train.log_every_n_steps=20 train.run_name=bert-warmup-tafeng-smoke-pt`

W&B run:
- https://wandb.ai/kronpoz/importance-aware-nbr/runs/l2jd5bm4

Output dir:
- `outputs/2026-04-18/01-33-18`

Artifacts:
- `outputs/2026-04-18/01-33-18/bert_best.pt`
- `outputs/2026-04-18/01-33-18/bert_last.pt`
- `outputs/2026-04-18/01-33-18/bert_encoder_bundle_tafeng.pt`
- `outputs/2026-04-18/01-33-18/bert_sanity_checks.txt`

Metrics:
- best_val_mlm_loss: 9.53411
- word2vec_loaded: 15743
- word2vec_missing: 0
- adjacent_cos: 0.965391
- random_cos: 0.961813
- adj_minus_rand: 0.003578

## Next Task
T3.1 — Run full plain BERT warmup for all datasets with realistic epochs (K) and log results:
- `data=tafeng`
- `data=instacart`
- `data=dunnhumby`

Then hand off saved encoder bundle checkpoints to GPT collaborator.

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

## Baseline Metrics (frequency)
- Full-dataset frequency baseline metrics and GPTopFreq alpha sweeps logged in `results/baseline_eval.md`
