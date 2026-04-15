# Context

## Current State
Phase 3 — Training scaffolding started. All Phase 0 (T0.1-T0.4), Phase 1 (T1.1-T1.5), and Phase 2 model components (T2.1-T2.4 embeddings, encoder, GPT, vanilla model) are done. Data preprocessing was run on all three datasets with the currently available files. Frequency baselines and evaluation script are implemented; full-dataset metrics and GPTopFreq alpha sweeps are logged in `results/baseline_eval.md` (subset runs retained for reference). Training scripts and Lightning modules are now scaffolded; no training runs executed yet.

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
- T2.6: added — bert_gpt vanilla model (CLS basket repr + GPT + dot-product logits)

**Phase 2 — Baselines (Frequency):**
- T2.5: started — nbr/baselines/frequency.py and scripts/evaluate_baselines.py created (evaluation needs runtime tuning)

**Phase 2.5 — Pre-training:**
- T2.7: Added `scripts/train_word2vec.py` to generate pre-trained item embeddings from basket data.
- T2.8: Completed Word2Vec pre-training for all datasets. Embeddings are stored as `word2vec_dim128.kv` files, ready for model initialization.

**Results:**
- results/baseline_eval.md — 2k-user subset metrics for Instacart, TaFeng, Dunnhumby (full runs pending)

## Next Task
T3 — Implement training loops for vanilla/dual-stream/full and add experiment configs; start training runs when requested.

## Decisions Made
- Paper: "Not All Items Are Created Equal: Importance-Aware Next Basket Recommendation"
- Three datasets: Instacart, Dunnhumby, TaFeng
- Three model tiers: vanilla → dual_stream → full
- CPU-only PyTorch installed (torch 2.11.0+cpu)
- Training loops should favor modular scripts with heavy W&B logging; PyTorch Lightning is allowed/preferred when it reduces boilerplate
- Hyperparameter tuning should use W&B Sweeps first; add other optimizers (e.g., Optuna) only if needed
- All preprocessing logic in single scripts/preprocess.py with shared helper
- BasketSequenceDataset: one sample per user, target = last basket, input = up to max_seq_len historical baskets
- ItemEmbedding.weight accessed via self.item_embedding.embedding.weight
- TaFeng file name is `ta_feng_all_months_merged.csv`
- Dunnhumby sample uses `transactions_*.csv` with columns CUST_CODE, BASKET_ID, SHOP_DATE, PROD_CODE
- Baseline evaluation uses basket-level histories from train split; test targets are last basket per user
- Baseline results are tracked in `results/` with command provenance
- Collaborators must update CHANGELOG/CONTEXT/INSTRUCTIONS before pushing
- Dataset schema and stats are documented in data/dataset_description.md
- Metadata tables (items/basket_meta/basket_items/user_meta) are now emitted during preprocessing
- Processed data will be distributed via a zip; raw data not required once processed is available

## Broken / Incomplete
- Dual-stream/full models are stubbed with vanilla backbone; importance losses still need real implementation
- Need to create experiments/ directory and write experiment configs
- Need to create tests/ directory and write unit tests

## Best Val Metrics
None yet.

## Baseline Metrics (full datasets)
- Logged full-dataset frequency baseline metrics and GPTopFreq alpha sweeps in results/baseline_eval.md
