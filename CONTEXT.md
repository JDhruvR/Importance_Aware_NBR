# Context

## Current State
Phase 2 — Baselines completed. All Phase 0 (T0.1-T0.4), Phase 1 (T1.1-T1.5), and Phase 2 model components (T2.1-T2.4 embeddings, encoder, GPT, vanilla model) are done. Data preprocessing was run on all three datasets with the currently available files. Frequency baselines and evaluation script are implemented; subset baseline metrics are logged in `results/baseline_eval.md`.

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
- T1.4: nbr/data/split.py — split_user_baskets() with last=val, second-to-last=test, rest=train
- T1.5: nbr/data/dataset.py — BasketSequenceDataset and BasketCollator

**Phase 2 — Baselines (Model Components):**
- T2.1: nbr/models/embeddings.py — ItemEmbedding + Word2VecTrainer
- T2.2: nbr/models/encoder.py — IntraBasketEncoder with CLS token, TransformerEncoderLayer
- T2.3: nbr/models/gpt.py — RoPEAttention + CausalBasketGPT (causality verified)
- T2.4: nbr/models/vanilla.py — VanillaNBR (embeddings -> encoder mean-pool -> GPT -> dot-product logits)

**Phase 2 — Baselines (Frequency):**
- T2.5: started — nbr/baselines/frequency.py and scripts/evaluate_baselines.py created (evaluation needs runtime tuning)

**Results:**
- results/baseline_eval.md — 2k-user subset metrics for Instacart, TaFeng, Dunnhumby (full runs pending)

## Next Task
T2.5 — Frequency baselines: finish evaluation runtime on full datasets and verify baseline metrics.

## Decisions Made
- Paper: "Not All Items Are Created Equal: Importance-Aware Next Basket Recommendation"
- Three datasets: Instacart, Dunnhumby, TaFeng
- Three model tiers: vanilla → dual_stream → full
- CPU-only PyTorch installed (torch 2.11.0+cpu)
- All preprocessing logic in single scripts/preprocess.py with shared helper
- BasketSequenceDataset: one sample per user, target = last basket, input = up to max_seq_len historical baskets
- ItemEmbedding.weight accessed via self.item_embedding.embedding.weight
- TaFeng file name is `ta_feng_all_months_merged.csv`
- Dunnhumby sample uses `transactions_*.csv` with columns CUST_CODE, BASKET_ID, SHOP_DATE, PROD_CODE
- Baseline evaluation uses basket-level histories from train split; test targets are last basket per user
- Baseline results are tracked in `results/` with command provenance
- Collaborators must update CHANGELOG/CONTEXT/INSTRUCTIONS before pushing

## Broken / Incomplete
- Need to run baseline evaluation on full datasets (subset runs done)
- Need to run GPTopFreq alpha sweeps
- Need to create experiments/ directory and training loops
- Need to create tests/ directory and write unit tests

## Best Val Metrics
None yet.

## Baseline Metrics (subset runs)
- Logged 2k-user subset results in results/baseline_eval.md
