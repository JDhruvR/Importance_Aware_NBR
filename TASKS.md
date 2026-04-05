# Tasks

Each task is self-contained and ends with a clear done condition. Tasks within a phase may be
parallelized unless a dependency is noted. All code lives under `nbr/` unless the task
explicitly says `scripts/` or `experiments/`.

---

## Phase 0 — Repo Bootstrap

### T0.1 — Initialize repo and tooling

- Run `uv init nbr` and configure `pyproject.toml`.
- Add all dependencies under `[project.dependencies]`:
  `torch`, `torchvision`, `einops`, `hydra-core`, `omegaconf`,
  `wandb`, `loguru`, `polars`, `numpy`, `faiss-cpu`, `gensim`,
  `pytest`, `ruff`.
- Add `[tool.ruff]` section: `line-length = 100`, enable `E`, `F`, `I` rule sets.
- Add a `Makefile` with targets: `lint`, `test`, `format`.
- **Done:** `uv run ruff check .` and `uv run pytest` both pass on an empty repo.

### T0.2 — Seed utility

Create `nbr/utils/seed.py`:
- Function `seed_everything(seed: int) -> None` that seeds `random`, `numpy`, `torch`,
  and `torch.cuda` (if available).
- **Done:** calling the function twice with the same seed produces the same model init.

### T0.3 — Device utility

Create `nbr/utils/device.py`:
- Function `get_device() -> torch.device` that returns CUDA if available, then MPS,
  then CPU.
- **Done:** runs correctly on a CPU-only machine.

### T0.4 — Hydra config skeleton

Create `configs/` with placeholder YAML files:
- `configs/data/instacart.yaml`, `configs/data/dunnhumby.yaml`, `configs/data/tafeng.yaml`
  (paths, filter thresholds).
- `configs/model/vanilla.yaml`, `configs/model/dual_stream.yaml`, `configs/model/full.yaml`
  (D, L1, L2, dk, K1, K2, num_heads).
- `configs/train/default.yaml` (lr, batch_size, epochs, warmup_steps, loss weights λ, γ, η).
- `configs/config.yaml` — top-level defaults list composing one of each.
- **Done:** `uv run python -c "import hydra"` works and a dummy launcher prints the composed config.

---

## Phase 1 — Data

### T1.1 — Download scripts

Create `scripts/download_data.py`:
- Downloads and extracts the following datasets to `data/raw/`:
  - **Instacart 2017:** https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
    (document the kaggle CLI command; do not automate auth).
  - **Dunnhumby "The Complete Journey":** https://www.dunnhumby.com/source-files/
  - **TaFeng:** https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset
- Print download instructions and expected file layout if the files are absent.
- **Done:** script prints a clear error message (not a stack trace) if files are missing.

### T1.2 — Instacart preprocessor

Create `scripts/preprocess.py` with a per-dataset preprocessing function for Instacart:

Input files: `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`.

Processing steps:
1. Merge prior and train splits; sort each user's orders by `order_number`.
2. Build `(user_id, order_sequence_index, item_id)` long table.
3. Filter users with fewer than 3 orders.
4. Filter items appearing fewer than 5 times in training baskets.
5. Remap user IDs and item IDs to contiguous integers starting at 0.
6. Save integer-remapped ID maps as `data/processed/instacart/item2id.json` and
   `user2id.json`.
7. Save the basket sequence table as `data/processed/instacart/baskets.parquet` with
   columns `[user_id: i32, order_idx: i32, item_id: i32]`.

- **Done:** parquet file loads without error, user count and item count match expected
  dataset statistics (printed to stdout).

### T1.3 — Dunnhumby and TaFeng preprocessors

Same structure as T1.2, adapted to each dataset's schema. Each writes to its own
subdirectory under `data/processed/`. The core logic (filter, remap, save) is extracted
into a shared `_preprocess_basket_df(df, min_baskets, min_item_freq)` helper so the three
dataset functions share it.

- **Done:** all three datasets produce parquet files with the same schema.

### T1.4 — Dataset split

Create `nbr/data/split.py`:
- Function `split_user_baskets(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]`
  returning train / val / test DataFrames.
- Split rule: for each user, last order → test, second-to-last → val, remainder → train.
  Users with exactly 3 orders get 1 train basket, 1 val basket, 1 test basket.
- **Done:** union of the three splits equals the full DataFrame; no user appears in two splits.

### T1.5 — PyTorch Dataset and collator

Create `nbr/data/dataset.py`:

`BasketSequenceDataset(torch.utils.data.Dataset)`:
- `__init__` takes the train DataFrame and a `max_seq_len: int`.
- `__getitem__(idx)` returns a dict with:
  - `item_seqs`: list of lists of item IDs (the user's basket sequence, up to `max_seq_len`
    most recent baskets, target basket excluded).
  - `target_items`: list of item IDs for the next basket.
  - `user_id`: int.

`BasketCollator`:
- Pads `item_seqs` to uniform basket count and uniform basket size within the batch.
- Returns a dict of tensors:
  - `items`: `(B, T, S)` int64 — item IDs per basket per timestep, 0-padded.
  - `basket_mask`: `(B, T)` bool — True for real baskets, False for padding.
  - `item_mask`: `(B, T, S)` bool — True for real items, False for padding.
  - `target`: `(B, V)` float32 — multi-hot ground truth.
  - `user_ids`: `(B,)` int64.
- **Done:** a batch from a DataLoader has the above shapes and all masks are correct.

---

## Phase 2 — Baselines

### T2.1 — Item embedding module

Create `nbr/models/embeddings.py`:

`ItemEmbedding(nn.Module)`:
- `__init__(vocab_size: int, dim: int)`.
- `forward(x: Tensor) -> Tensor` — lookup, returns same shape with extra `D` dimension.
- `from_word2vec(path: str, vocab_size: int, dim: int) -> ItemEmbedding` — class method
  that loads a gensim KeyedVectors file and initializes weights.

`Word2VecTrainer`:
- `train(basket_sequences: list[list[int]], dim: int, window: int, epochs: int) -> KeyedVectors`
  — wraps gensim `Word2Vec`, treats each basket as a sentence.
- **Done:** `ItemEmbedding` forward produces shape `(*, D)` for any input shape `(*)`.

### T2.2 — BERT-style intra-basket encoder (vanilla)

Create `nbr/models/encoder.py`:

`IntraBasketEncoder(nn.Module)`:
- `__init__(dim: int, num_heads: int, num_layers: int, dropout: float)`.
- Prepends a learned CLS token to each basket sequence.
- Applies `num_layers` of `nn.TransformerEncoderLayer` with `batch_first=True`.
- No positional encoding (baskets are unordered sets).
- `forward(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]`
  - Input `x`: `(B*T, S, D)` — item embeddings for all baskets in the batch, flattened
    across time.
  - Input `mask`: `(B*T, S)` bool — True for real items.
  - Returns `(cls_repr, item_reprs)`: `(B*T, D)` and `(B*T, S, D)`.
- Internally prepend CLS, run transformer, split outputs.
- **Done:** shape test passes; CLS output is different from the mean of item outputs.

### T2.3 — Causal GPT inter-basket encoder (vanilla)

Create `nbr/models/gpt.py`:

`RoPEAttention(nn.Module)`:
- Multi-head attention with Rotary Position Embeddings applied to Q and K only.
- Implements the rotation as described in the RoFormer paper: split head dim into pairs,
  apply cos/sin rotation with position index.
- `forward(x: Tensor, causal_mask: Tensor) -> Tensor`.

`CausalBasketGPT(nn.Module)`:
- `__init__(dim: int, num_heads: int, num_layers: int, dropout: float)`.
- Stack of `num_layers` blocks, each: `RoPEAttention` → residual → LayerNorm →
  FFN → residual → LayerNorm.
- Causal mask prevents basket $t$ from attending to basket $t' > t$.
- `forward(basket_reprs: Tensor, basket_mask: Tensor) -> Tensor`
  - Input `basket_reprs`: `(B, T, D)`.
  - Returns `(B, T, D)` — predicted next-basket representations at each position.
- **Done:** with a batch of 2 sequences of length 5, output at position `t` does not change
  when baskets at positions `> t` are modified (verified by a unit test).

### T2.4 — Vanilla NBR model

Create `experiments/01_vanilla.py` and `nbr/models/vanilla.py`:

`VanillaNBR(nn.Module)`:
- Combines `ItemEmbedding` → `IntraBasketEncoder` (mean pool over items, no CLS) →
  `CausalBasketGPT` → dot product against `ItemEmbedding.weight` → logits over vocab.
- `forward(items, item_mask, basket_mask) -> Tensor` returns `(B, T, V)` logits.
- Loss: `F.binary_cross_entropy_with_logits` on the multi-hot target.

Experiment launcher `experiments/01_vanilla.py`:
- Loads config via Hydra.
- Trains with AdamW and a linear warmup + cosine decay schedule.
- Evaluates every epoch on val set; saves best checkpoint by Recall@10.
- Logs all metrics to wandb.
- **Done:** training loop runs to completion on a small subset; val Recall@10 is logged.

### T2.5 — Frequency baselines

Create `nbr/baselines/frequency.py`:

- `GlobalTopFreq(topk: int)` — recommends globally most frequent items.
- `PersonalTopFreq(topk: int)` — recommends user's personally most frequent items.
- `GPTopFreq(topk: int, alpha: float)` — GP-TopFreq hybrid from Li et al. 2023.

All implement a common `predict(user_history: list[list[int]]) -> list[int]` interface.

Create `scripts/evaluate_baselines.py` that runs all three on the test split and prints
a table of Recall@K, Repeat Recall@K, Explore Recall@K for K ∈ {5, 10, 20}.

- **Done:** GP-TopFreq scores are within expected range of Li et al. reported numbers.

---

## Phase 3 — Importance Scoring

### T3.1 — Geometric importance score computation

Create `scripts/compute_importance.py`:

- Loads a trained `IntraBasketEncoder` checkpoint (from Phase 2 warmup).
- For each training basket, computes `delta(i, B_t)` for every item `i` by running a
  forward pass with item `i` removed and measuring the L2 shift in `cls_repr`.
- Normalizes within each basket, averages across appearances, applies IDF correction:
  `alpha_idf[i] = delta_bar[i] * log(N / df[i])`.
- Saves `alpha_idf` as `data/processed/{dataset}/importance_scores.npy` (shape `(V,)`).
- Prints histogram statistics: mean, std, 25th/50th/75th percentile.
- **Done:** scores are saved and non-uniform (variance > 0 across items).

### T3.2 — Importance head module

Create `nbr/models/importance.py`:

`ImportanceHead(nn.Module)`:
- `__init__(dim: int)`.
- Two-layer MLP: `Linear(D, D//2)` → `GELU` → `Linear(D//2, 1)` → `Sigmoid`.
- `forward(item_reprs: Tensor) -> Tensor`
  - Input: `(B*T, S, D)`.
  - Output: `(B*T, S)` importance weights in `[0, 1]`.

`importance_init_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor`:
- MSE loss between predicted importance weights and `alpha_idf` target scores,
  masked to real items only.
- **Done:** shape test passes; loss decreases when weights are pushed toward targets.

---

## Phase 4 — Dual-Stream Fusion

### T4.1 — Gated basket fusion module

Create `nbr/models/fusion.py`:

`DualStreamFusion(nn.Module)`:
- `__init__(dim: int)`.
- `forward(cls_repr: Tensor, item_reprs: Tensor, importance: Tensor, item_mask: Tensor) -> Tensor`
  - `cls_repr`: `(B*T, D)` — full basket summary.
  - `item_reprs`: `(B*T, S, D)`.
  - `importance`: `(B*T, S)` — from `ImportanceHead`.
  - `item_mask`: `(B*T, S)` bool.
  - Computes `basket_core` as importance-weighted mean over real items.
  - Computes elementwise gate `g = sigmoid(W_g([basket_full; basket_core]))`.
  - Returns `g * basket_full + (1 - g) * basket_core`: `(B*T, D)`.
- `W_g` is `nn.Linear(2*D, D, bias=True)`.
- **Done:** output shape is `(B*T, D)`; when all importance weights are equal the output
  is close to (but not identical to) mean pool, since the gate is also learned.

### T4.2 — Dual-stream model and experiment

Create `nbr/models/dual_stream.py` and `experiments/02_dual_stream.py`:

`DualStreamNBR(nn.Module)`:
- Same as `VanillaNBR` but replaces mean pool with `IntraBasketEncoder` (with CLS) +
  `ImportanceHead` + `DualStreamFusion`.
- Training schedule:
  - **Phase 1 (warmup):** train encoder + GPT + fusion with uniform BCE; importance head
    frozen.
  - **Phase 2 (importance init):** freeze encoder; compute `alpha_idf`; pre-train
    `ImportanceHead` with `importance_init_loss`.
  - **Phase 3 (joint):** unfreeze all; train with
    `L = L_BCE + eta * L_MLM` (no two-stage decoder yet).
- MLM auxiliary loss: randomly mask 15% of items per basket; predict them from context
  via `Linear(D, V)` applied to the masked item's `h_i`.
- **Done:** Recall@10 on val set exceeds `VanillaNBR` by at least 0.5pp, confirming
  dual-stream fusion adds signal.

---

## Phase 5 — Full Model

### T5.1 — Orthogonal intent-fill decomposition

Create `nbr/models/projection.py`:

`IntentProjection(nn.Module)`:
- `__init__(dim: int, intent_dim: int)`.
- `P`: `nn.Parameter` of shape `(D, dk)`, initialized with `nn.init.orthogonal_`.
- `forward(x: Tensor) -> tuple[Tensor, Tensor]`
  - Input `x`: `(B, D)`.
  - Returns `(intent_repr, fill_repr)`:
    - `intent_repr = x @ P @ P.T`: `(B, D)`.
    - `fill_repr = x - intent_repr`: `(B, D)`.
- `orthogonalize_()` method: in-place Gram-Schmidt re-orthonormalization of `P`'s columns.
  Call this every N steps from the training loop.
- `orthogonality_loss(intent_repr, fill_repr) -> Tensor`:
  - `(intent_repr * fill_repr).sum(dim=-1).pow(2).mean()`.
- **Done:** after `orthogonalize_()`, `P.T @ P` is close to identity (max abs deviation < 1e-5).

### T5.2 — Two-stage conditioned decoder

Create `nbr/models/decoder.py`:

`TwoStageDecoder(nn.Module)`:
- `__init__(dim: int, intent_dim: int, temperature: float)`.
- Holds `IntentProjection` and a conditioning linear `W_cond: nn.Linear(D, D)`.
- `forward(next_pred: Tensor, item_embeddings: Tensor, core_mask: Tensor | None) -> dict`
  - During training (`core_mask` is None):
    1. Decompose `next_pred` into `intent_repr`, `fill_repr`.
    2. Compute intent logits: `(V, D) @ (B, D).T → (B, V)`.
    3. Compute soft intent context:
       `soft_intent = softmax(intent_logits / tau) @ item_embeddings`: `(B, D)`.
    4. Compute conditioned fill query:
       `fill_query = LN(fill_repr + W_cond(soft_intent))`.
    5. Compute fill logits: `(V, D) @ (B, D).T → (B, V)`.
    6. Return `{"intent_logits": (B, V), "fill_logits": (B, V), "soft_intent": (B, D)}`.
  - At inference (`core_mask` provided — bool tensor marking predicted core items):
    - Compute `hard_intent` as mean of predicted core item embeddings.
    - Compute conditioned fill query with `hard_intent` instead of `soft_intent`.
    - Return same dict.

`residual_decode(repr: Tensor, item_embeddings: Tensor, projection: Tensor | None, k: int, excluded: set) -> list[int]`:
- Pure function (no `nn.Module`), no gradient.
- Implements the residual loop from Section 6 of the paper.
- `projection`: if provided, project item embeddings before subtraction (for the intent
  subspace: `P @ P.T @ e_i`; for fill: `(I - P @ P.T) @ e_i`).
- Returns list of `k` item IDs.
- **Done:** calling twice with the same inputs returns the same list; items in `excluded`
  never appear in output.

### T5.3 — Full model and loss

Create `nbr/models/full_model.py`:

`IntentAwareNBR(nn.Module)`:
- Composes all prior modules:
  `ItemEmbedding` → `IntraBasketEncoder` → `ImportanceHead` → `DualStreamFusion` →
  `CausalBasketGPT` → `TwoStageDecoder`.
- `forward(items, item_mask, basket_mask, alpha_idf_targets) -> dict`
  - Returns dict with keys: `intent_logits`, `fill_logits`, `soft_intent`, `importance`,
    `cls_repr`, `masked_item_logits`.

Create `nbr/losses.py`:

- `intent_loss(logits, targets, alpha_idf)` — importance-weighted BCE per equation 11.
- `fill_loss(logits, fill_targets)` — uniform BCE on fill targets.
- `total_loss(...)` — combines all four losses with configurable weights.

- **Done:** `total_loss` output is a scalar; all gradients flow back to `ItemEmbedding.weight`.

### T5.4 — Full model experiment

Create `experiments/03_full_model.py`:

Same structure as `02_dual_stream.py` but with `IntentAwareNBR` and the four-phase
training schedule from Section 5.8 of the paper. Key additions:
- Phase 3 uses `importance_init_loss`; Phase 4 uses full `total_loss`.
- Call `intent_proj.orthogonalize_()` every 100 steps.
- Log `intent_loss`, `fill_loss`, `orth_loss`, `mlm_loss` as separate wandb metrics.
- After training, run the inference procedure from Section 6 with `residual_decode`.
- **Done:** Recall@10 on val exceeds `DualStreamNBR`.

---

## Phase 6 — Evaluation and Metrics

### T6.1 — Metrics module

Create `nbr/metrics.py`:

All functions accept `predicted: list[int]` and `ground_truth: list[int]`:
- `recall_at_k(predicted, ground_truth, k: int) -> float`.
- `ndcg_at_k(predicted, ground_truth, k: int) -> float`.
- `hit_rate_at_k(predicted, ground_truth, k: int) -> float`.
- `repeat_recall_at_k(predicted, ground_truth, user_history_items: set[int], k) -> float`
  — restricts ground truth to items seen in user history.
- `explore_recall_at_k(predicted, ground_truth, user_history_items: set[int], k) -> float`
  — restricts ground truth to items NOT seen in user history.

`evaluate_model(model, dataloader, k_values: list[int]) -> dict[str, float]`:
- Runs inference for each user; collects per-user predictions.
- Returns a flat dict: `{"recall@5": ..., "repeat_recall@10": ..., ...}` for all metrics
  and all K values.

- **Done:** `recall_at_k([1,2,3], [2,4], 3) == 0.5`; test case for each function.

### T6.2 — Final evaluation script

Create `scripts/evaluate.py`:

- Loads a checkpoint for each of the three model variants.
- Runs `evaluate_model` on the test split.
- Prints a LaTeX-formatted table of all metrics across all models and K values.
- **Done:** table renders cleanly and numbers are reproducible across two runs with the
  same seed.

### T6.3 — Ablation table

Create `scripts/ablation.py`:

Trains and evaluates the following five configurations on Instacart, saving results to
`results/ablation.json`:

| Config | Importance head | Dual-stream fusion | Two-stage decoder |
|---|---|---|---|
| Vanilla | ✗ | ✗ | ✗ |
| +Importance | ✓ | ✗ | ✗ |
| +DualStream | ✓ | ✓ | ✗ |
| +Intent decomp | ✓ | ✓ | Intent only |
| Full | ✓ | ✓ | ✓ |

- **Done:** `ablation.json` contains all five rows; each row has Recall@10, Repeat Recall@10,
  Explore Recall@10.

---

## Phase 7 — Polish

### T7.1 — README

Write `README.md` covering:
- One-paragraph description of the problem and approach.
- Setup: `uv sync`, dataset download instructions.
- How to run each experiment (`uv run python experiments/01_vanilla.py`).
- How to reproduce the ablation table.
- Table of results (fill in after T6.3).

### T7.2 — Unit test suite

Write tests in `tests/` covering at minimum:
- `test_encoder.py`: shape test for `IntraBasketEncoder`; causality test for `CausalBasketGPT`.
- `test_projection.py`: orthogonality of `P` after `orthogonalize_()`; sum-to-input property
  of decomposition.
- `test_decoder.py`: no excluded items in output of `residual_decode`; no duplicate items.
- `test_metrics.py`: hand-computed expected values for each metric function.
- `test_collator.py`: mask correctness for a small batch with unequal basket sizes.

- **Done:** `uv run pytest` passes with zero failures.

### T7.3 — Inference speed benchmark

Create `scripts/benchmark_inference.py`:
- Times the full inference pipeline (Steps 1–6 from Section 6) on a batch of 64 users,
  averaged over 10 runs.
- Reports: BERT encode time, GPT forward time, core decode time, fill decode time, total time.
- **Done:** total inference time per user is printed; FAISS index is used for catalog lookup.
