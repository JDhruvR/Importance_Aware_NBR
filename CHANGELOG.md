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

## 2026-04-06 — Dataset preprocessing aligned to actual files
- Changed: scripts/preprocess.py to handle TaFeng merged file name and Dunnhumby sample files
- Changed: scripts/preprocess.py to select and cast required Dunnhumby columns consistently
- Changed: .gitignore to only ignore data directories (not all csv/json/parquet globally)
- Changed: INSTRUCTIONS.md with human download steps for all three datasets

## 2026-04-06 — Frequency baselines scaffolding
- Added: nbr/baselines/frequency.py with GlobalTopFreq, PersonalTopFreq, GPTopFreq
- Added: scripts/evaluate_baselines.py for Recall/Repeat/Explore metrics
- Changed: nbr/data/split.py to remove invalid user-overlap assertion

## 2026-04-06 — Baseline evaluation runtime
- Changed: scripts/evaluate_baselines.py to use aggregated counts (faster on large datasets)
- Added: --max-users flag for subset evaluation to keep runtime manageable
- Done: baseline metrics computed on 2k-user subsets for Instacart, TaFeng, Dunnhumby

## 2026-04-06 — Baseline results tracking
- Added: results/baseline_eval.md with commands and subset metrics for all datasets (full runs pending)

## 2026-04-06 — Collaboration hygiene
- Changed: INSTRUCTIONS.md to require changelog/context/results updates before push

## 2026-04-06 — BERT+GPT vanilla model
- Added: nbr/models/bert_gpt/model.py with BertGptNBR (CLS basket repr + GPT + dot logits)
- Added: nbr/models/bert_gpt/__init__.py to expose BertGptNBR

## 2026-04-06 — Dataset documentation
- Added: data/dataset_description.md with schema, source files, and stats for all datasets

## 2026-04-06 — Metadata tables in preprocessing
- Changed: scripts/preprocess.py to emit items/basket_meta/basket_items/user_meta tables
- Changed: scripts/preprocess.py to filter missing Dunnhumby customers and align columns

## 2026-04-06 — README data setup
- Added: README.md with processed data zip instructions (raw data optional)

## 2026-04-06 — Full baseline evaluation
- Changed: results/baseline_eval.md with full-dataset metrics and GPTopFreq alpha sweeps
- Changed: CONTEXT.md snapshot updated after baseline evaluation completion

## 2026-04-06 — Training workflow guidance
- Changed: INSTRUCTIONS.md to clarify training loop and HPO tooling preferences

## 2026-04-06 — Training scaffolding
- Added: nbr/train/data_module.py for Lightning dataloaders
- Added: nbr/train/vanilla_lightning.py Lightning module for vanilla/BERT+GPT
- Added: nbr/metrics/ranking.py with Recall/NDCG and repeat/explore helpers
- Added: nbr/utils/logger.py for loguru and W&B environment setup
- Added: scripts/train_vanilla.py Lightning training entrypoint
- Changed: configs/train/default.yaml with trainer/logging settings
- Changed: configs/model/vanilla.yaml to include model_type
- Added: nbr/train/__init__.py and nbr/metrics/__init__.py
- Changed: pyproject.toml to include lightning dependency

## 2026-04-06 — Dual-stream/full training scaffolding
- Added: nbr/train/importance_lightning.py for importance-aware Lightning training
- Added: nbr/models/dual_stream.py and nbr/models/full.py placeholders with vanilla backbone
- Added: scripts/train_importance.py training entrypoint for dual-stream/full
- Changed: configs/model/dual_stream.yaml and configs/model/full.yaml to include model_type
- Changed: nbr/models/__init__.py to export new model stubs

## 2026-04-15 — Word2Vec pre-training script
- Added: scripts/train_word2vec.py to pre-train item embeddings on basket data.
- Changed: The script dynamically sets the window size to the max basket length per dataset.
- Done: Trained embeddings for all 3 datasets and saved as `.kv` files in `data/processed/{dataset}/`.
- Added: `scripts/check_embeddings.py` to validate trained Word2Vec embeddings by finding similar items.