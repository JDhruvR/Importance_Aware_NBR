# Instructions

## Project Overview

Implement and validate the intent-aware Next Basket Recommendation system from
"Not All Items Are Created Equal: Importance-Aware Next Basket Recommendation".
The repo progresses from clean baselines to the full architecture; each stage is
independently evaluable so ablations come for free.

---

## File Tracking Rules

- **Always track:** code files (.py), configs (.yaml, .toml), docs (.md, .tex), Makefile, .gitignore
- **Never track:** raw data, processed data (.parquet, .csv, .json), checkpoints (.pt, .pth), logs, outputs/, wandb/, caches (__pycache__, .ruff_cache, .venv), uv.lock
- **Tests:** keep major/structural test files in `tests/` and track them. Delete temporary/scratch test files after use.
- **Living docs:** CHANGELOG.md, CONTEXT.md, INSTRUCTIONS.md are tracked and updated every session.
- **When in doubt:** ask before adding something to .gitignore or committing.

---

## Collaboration Hygiene (Required)

Before any push:
- Update `CHANGELOG.md` with all meaningful changes (one bullet per file/module touched).
- Overwrite `CONTEXT.md` with a precise snapshot of the current state.
- Update `INSTRUCTIONS.md` if new decisions or workflows are introduced.
- Add results to `results/` in a clean, human-readable format whenever a run produces metrics.

All collaborators must follow this so downstream work is predictable and resumable.

## Keep This File Compact

Target length: ~200 lines. If it grows beyond that, consolidate or move detail to
other docs (e.g., `CHANGELOG.md` or `results/`). Reduce length before pushing.

---

## Dataset Downloads (Human Instructions)

These are manual steps for humans. Do not automate authentication.

### Instacart 2017
Source: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

1. Create Kaggle API token from your account settings.
2. Download the dataset and unzip into: `data/raw/instacart/`
3. Required files:
   - `orders.csv`
   - `order_products__prior.csv`
   - `order_products__train.csv`

### TaFeng
Source: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset

1. Download the dataset and unzip into: `data/raw/tafeng/`
2. Expected file:
   - `ta_feng_all_months_merged.csv`

### Dunnhumby (sample 50k customers)
Source: dunnhumby Complete Journey (sample ZIP)

1. Download the sample ZIP and extract into: `data/raw/dunnhumby/`
2. Expected files:
   - `transactions_*.csv` (multiple monthly files)
   - `time.csv`

---

## Living Documents — Maintain These Every Session

### `changelog.md`
One entry per meaningful change. Format:

```
## YYYY-MM-DD — <short title>
- Added: <what and why>
- Changed: <what and why>
- Deleted: <what and why>
```

Rules: one bullet per file/module touched. No prose. If a change spans multiple
files for one logical reason, group them under one entry. Append only — never edit
past entries. Keep entries concise (aim for 3–6 bullets each).

### `context.md`
A single snapshot of *where the project is right now*. **Overwrite it entirely** after
every few tasks or at any natural stopping point. It is the file a new agent or
collaborator reads first to get up to speed. It must answer:

- What phase are we in and what was just completed?
- What is the next task (T-code) and what does it need?
- What decisions were made that aren't obvious from the code?
- What is currently broken or incomplete?
- What are the current best val metrics (model, dataset, Recall@10)?

Target length: 100–200 lines. Enough detail to resume without reading the full codebase.
No speculation — only facts about the current state.

**Update `changelog.md` for every change. Update `context.md` at the end of every
work session or after completing a phase.**

---

## Environment

- **Package manager:** `uv` exclusively.
- **Python:** 3.11+. **Framework:** PyTorch 2.x.
- **Config:** `hydra-core`. Every hyperparameter lives in `configs/`. Models and
  trainers accept plain Python types, never config objects.
- **Logging:** `wandb` for metrics, `loguru` for console.
- **Training loops:** PyTorch Lightning is allowed/preferred when it reduces boilerplate; keep modules and callbacks modular.
- **HPO:** prefer W&B Sweeps for hyperparameter tuning; add other optimizers (e.g., Optuna) only if needed.
- **Data:** `polars` for preprocessing, `numpy` for array ops outside PyTorch.
- **Nearest neighbor:** `faiss-cpu` (or `faiss-gpu`) at inference.
- **Reproducibility:** single `utils/seed.py` helper seeds Python, NumPy, PyTorch, CUDA.

---

## Repository Layout

```
configs/   data/   nbr/   scripts/   experiments/   tests/   results/
```

---

## Code Style

- **Readable over clever.** If a line needs a comment to be understood, rewrite it.
- **Type-annotated everywhere.** Full type hints on every function. `from __future__ import annotations` at the top of every file.
- **No magic numbers** outside config files.
- **Imports:** stdlib → third-party → local, separated by blank lines. Absolute only.
- **Line length:** 100. **Formatter:** `ruff format`. **Linter:** `ruff check`. Both pass clean before any commit.

### PyTorch
- Shape comment on first use if non-obvious: `# (B, T, D)`.
- Never `.cuda()` directly — resolve device from config or `utils/device.py`.
- `nn.Module`: `__init__` declares submodules only; `forward` is pure computation.
- One class per concept.
- Use `einops` for non-trivial reshaping over chains of `.view()` / `.permute()`.

### Comments
Write only when: (1) a non-obvious algorithmic choice needs one line of justification,
(2) a shape annotation aids clarity, (3) a training phase boundary is entered.
Never restate what the code obviously does.

### Error Handling
- No `try/except` for control flow.
- `assert` for invariants (tensor shapes, config consistency).
- `ValueError` / `RuntimeError` with a descriptive message for user-facing misconfiguration.

---

## Model Conventions

| Paper symbol | Python name |
|---|---|
| `h_CLS` | `cls_repr` |
| `alpha_i` | `importance` |
| `b_t^full / ^core` | `basket_full / basket_core` |
| `b_t` | `basket_repr` |
| `hat_h_{T+1}` | `next_pred` |
| `hat_h^intent / ^fill` | `intent_repr / fill_repr` |
| `P` | `intent_proj` |
| `tilde_c` | `soft_intent` |

Dimension shorthands: `B` batch, `T` sequence, `S` items/basket, `D` hidden, `V` vocab, `dk` intent dim.

---

## Experiment Progression

Vanilla → Dual-stream → Full, same split and seed across experiments.

Do not run training jobs unless explicitly requested; implement scripts/configs first.

1.  **Preprocessing:** Run `scripts/preprocess.py` if starting from raw data.
2.  **Baselines:** Run `scripts/evaluate_baselines.py` to get frequency-based baseline metrics.
3.  **Pre-training:** Run `scripts/train_word2vec.py` for each dataset to generate item embeddings. These are saved as `.kv` files.
    ```bash
    PYTHONPATH=. python scripts/train_word2vec.py --dataset <name>
    ```
4.  **Model Training:** Run `scripts/train_vanilla.py` and `scripts/train_importance.py` to train and evaluate the neural models, using the pre-trained embeddings.

---

## Evaluation Protocol

Report Recall@K, Repeat Recall@K, Explore Recall@K, NDCG@K for K ∈ {5, 10, 20}.
Split: last basket → test, second-to-last → val, rest → train.
Filter: users with < 3 baskets and items appearing < 5 times in training.
