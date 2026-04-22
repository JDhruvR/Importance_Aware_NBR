# BERT Warmup Results

Plain basket-BERT warmup (MLM only), no Lightning, W&B tracked.

## Smoke Run (TaFeng)

Command:

```bash
PYTHONPATH=. uv run python scripts/train_bert.py data=tafeng train.epochs=1 train.batch_size=64 train.num_workers=0 train.max_train_baskets=1500 train.max_val_baskets=300 train.log_every_n_steps=20 train.run_name=bert-warmup-tafeng-smoke-pt
```

W&B run:
- https://wandb.ai/kronpoz/importance-aware-nbr/runs/l2jd5bm4

Artifacts (local output dir `outputs/2026-04-18/01-33-18/`):
- `bert_best.pt`
- `bert_last.pt`
- `bert_encoder_bundle_tafeng.pt`
- `bert_sanity_checks.txt`

Sanity metrics:
- `word2vec_loaded`: 15743
- `word2vec_missing`: 0
- `best_val_mlm_loss`: 9.53411
- `adjacent_cos`: 0.965391
- `random_cos`: 0.961813
- `adj_minus_rand`: 0.003578

Notes:
- `adj_minus_rand` positive in smoke run, but small due to tiny subset and 1 epoch.
- Use full data + more epochs for meaningful representation quality judgment.

## Saved Run Config Snapshots

W&B-exported run config snapshots are stored under `results/run_configs/bert_warmup/` instead of `configs/`.

- `results/run_configs/bert_warmup/tafeng_2026-04-20_00-12-59.yaml`
- `results/run_configs/bert_warmup/dunnhumby_2026-04-20_00-56-51.yaml`
- `results/run_configs/bert_warmup/instacart_2026-04-22_15-15-10.yaml`

Operational note:
- After full runs, keep a copy of `bert_best.pt` and `bert_encoder_bundle_<dataset>.pt` under `data/processed/<dataset>/` on the training machine for quick downstream handoff.
