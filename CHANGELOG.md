# Changelog

## 2025-XX-XX — Project initialized
- Added: instructions.md and tasks.md in repo root

## 2026-04-05 — Initialize repo and tooling
- Added: pyproject.toml with all required dependencies
- Added: Makefile with lint, test, format targets
- Done: uv init completed
- In progress: Installing dependencies via uv pip install

## 2026-04-05 — Seed utility
- Created: nbr/utils/seed.py with seed_everything function
- Created: test_seed.py to verify functionality

## 2026-04-05 — Device utility verified
- Created: nbr/utils/device.py with get_device() function
- Done: get_device() returns cpu on CPU-only machine

## 2026-04-05 — Hydra config skeleton
- Created: configs/config.yaml, configs/data/{instacart,dunnhumby,tafeng}.yaml
- Created: configs/model/{vanilla,dual_stream,full}.yaml
- Created: configs/train/default.yaml
- Done: config composition verified, all files load and merge correctly

## 2026-04-05 — Download scripts
- Created: scripts/download_data.py with instructions for all 3 datasets
- Done: script prints clear error messages when files are missing

## 2026-04-05 — Preprocessing pipeline
- Created: scripts/preprocess.py with Instacart, Dunnhumby, TaFeng preprocessors
- Created: shared _preprocess_basket_df helper for filter/remap/save logic
- Done: all 3 datasets produce parquet with same schema [user_id: i32, order_idx: i32, item_id: i32]

## 2026-04-05 — Dataset split and PyTorch Dataset
- Created: nbr/data/split.py with split_user_baskets() function
- Created: nbr/data/dataset.py with BasketSequenceDataset and BasketCollator
- Created: nbr/data/__init__.py
- Done: split logic verified (union equals original, no user overlap)
- Done: collator produces correct shapes: items (B,T,S), basket_mask (B,T), item_mask (B,T,S), target (B,V)

## 2026-04-05 — Phase 2: Baselines (embeddings, encoder, GPT, vanilla model)
- Created: nbr/models/embeddings.py — ItemEmbedding + Word2VecTrainer
- Created: nbr/models/encoder.py — IntraBasketEncoder with CLS token
- Created: nbr/models/gpt.py — RoPEAttention + CausalBasketGPT with causality verified
- Created: nbr/models/vanilla.py — VanillaNBR combining all components
- Done: all shape tests pass, causality verified, gradients flow correctly