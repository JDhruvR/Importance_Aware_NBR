# Context

## Current State
Phase 2 — Baselines completed. All Phase 0 (T0.1-T0.4), Phase 1 (T1.1-T1.5), and Phase 2 model components (T2.1-T2.4 embeddings, encoder, GPT, vanilla model) are done.

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

## Next Task
T2.5 — Frequency baselines: Create nbr/baselines/frequency.py with GlobalTopFreq, PersonalTopFreq, GPTopFreq.

## Decisions Made
- Paper: "Not All Items Are Created Equal: Importance-Aware Next Basket Recommendation"
- Three datasets: Instacart, Dunnhumby, TaFeng
- Three model tiers: vanilla → dual_stream → full
- CPU-only PyTorch installed (torch 2.11.0+cpu)
- All preprocessing logic in single scripts/preprocess.py with shared helper
- BasketSequenceDataset: one sample per user, target = last basket, input = up to max_seq_len historical baskets
- ItemEmbedding.weight accessed via self.item_embedding.embedding.weight

## Broken / Incomplete
- No raw data downloaded yet — preprocessing and dataset classes cannot be end-to-end tested
- Need to create nbr/baselines/ directory and implement frequency baselines
- Need to create experiments/ directory and training loops
- Need to create tests/ directory and write unit tests

## Best Val Metrics
None yet.
