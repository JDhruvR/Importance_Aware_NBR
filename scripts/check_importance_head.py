"""Validate a trained ImportanceHead checkpoint.

Loads the frozen BERT encoder and trained ImportanceHead, runs a sample of
baskets through both, and performs the following checks:

  1. Checkpoint structure and model loadability
  2. Output range — all predictions in [0, 1]
  3. Target correlation — Pearson/Spearman between predicted and alpha_idf
  4. Per-item agreement — average predicted vs target for top/bottom items
  5. Context sensitivity — same item gets different scores in different baskets
  6. Ranking agreement — overlap between top-K predicted and top-K target items

Usage:
    python scripts/check_importance_head.py --dataset instacart --checkpoint outputs/importance_head/.../importance_head_best.pt
    python scripts/check_importance_head.py --dataset instacart --checkpoint outputs/importance_head/.../importance_head_best.pt --max-baskets 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats

from nbr.data.split import split_user_baskets
from nbr.models.encoder import IntraBasketEncoder
from nbr.models.importance import ImportanceHead
from nbr.utils.device import get_device

PROJECT_ROOT = Path(__file__).parent.parent

# ── colour helpers (ANSI) ────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET}: {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}: {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ WARN{RESET}: {msg}")


def _header(msg: str) -> None:
    print(f"\n{BOLD}{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}{RESET}")


# ── item name lookup (same as check_importance.py) ──────────────────
def _build_name_map(processed_dir: Path, dataset: str) -> dict[int, str]:
    items_path = processed_dir / "items.parquet"
    item_map_path = processed_dir / "item_map.parquet"

    if not items_path.exists():
        return {}

    items_df = pl.read_parquet(items_path)

    possible_name_cols = ["product_name", "product_description", "description"]
    if dataset == "dunnhumby":
        possible_name_cols.append("prod_code_raw")
    if dataset == "tafeng":
        possible_name_cols.append("product_subclass")

    name_col = next((c for c in possible_name_cols if c in items_df.columns), None)
    if name_col is None:
        return {}

    if item_map_path.exists():
        item_map_df = pl.read_parquet(item_map_path)
        orig_id_col = (
            "original_item_id" if "original_item_id" in items_df.columns else "item_id"
        )
        orig_to_name = dict(zip(items_df[orig_id_col], items_df[name_col], strict=True))
        new_to_orig = dict(
            zip(item_map_df["item_id"], item_map_df["original_item_id"], strict=True)
        )
        return {nid: str(orig_to_name.get(oid, "N/A")) for nid, oid in new_to_orig.items()}

    id_col = "item_id" if "item_id" in items_df.columns else "original_item_id"
    return {int(k): str(v) for k, v in zip(items_df[id_col], items_df[name_col], strict=True)}


# ── inference helper ─────────────────────────────────────────────────
def _run_inference(
    embedding: torch.nn.Embedding,
    encoder: IntraBasketEncoder,
    head: ImportanceHead,
    baskets: list[list[int]],
    item_id_offset: int,
    device: torch.device,
    max_items: int = 50,
    batch_size: int = 256,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Run baskets through frozen BERT + trained head.

    Returns:
        predictions: list of 1-D arrays, one per basket, with predicted scores.
        item_ids: list of item ID lists, one per basket.
    """
    head.eval()
    all_preds: list[np.ndarray] = []
    all_items: list[list[int]] = []

    with torch.no_grad():
        for i in range(0, len(baskets), batch_size):
            chunk = baskets[i : i + batch_size]
            max_len = max(min(len(b), max_items) for b in chunk)

            input_ids_list = []
            mask_list = []
            items_list = []

            for basket in chunk:
                items = basket[:max_items]
                S = len(items)
                pad = max_len - S
                ids = torch.tensor(
                    [iid + item_id_offset for iid in items], dtype=torch.long,
                )
                ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long)])
                mask = torch.cat([
                    torch.ones(S, dtype=torch.bool),
                    torch.zeros(pad, dtype=torch.bool),
                ])
                input_ids_list.append(ids)
                mask_list.append(mask)
                items_list.append(items)

            input_ids = torch.stack(input_ids_list).to(device)
            attention_mask = torch.stack(mask_list).to(device)

            token_emb = embedding(input_ids)
            _, item_reprs = encoder(token_emb, attention_mask)
            predicted = head(item_reprs)  # (B, S)

            for j, items in enumerate(items_list):
                S = len(items)
                scores = predicted[j, :S].cpu().numpy()
                all_preds.append(scores)
                all_items.append(items)

    return all_preds, all_items


# ── checks ───────────────────────────────────────────────────────────

def check_checkpoint(ckpt_path: Path, dim: int) -> dict | None:
    """Check 1: checkpoint loads and contains expected keys."""
    _header("Check 1 — Checkpoint structure")

    if not ckpt_path.exists():
        _fail(f"Checkpoint not found: {ckpt_path}")
        return None
    _pass(f"Checkpoint exists: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    expected_keys = {"head_state_dict", "dim", "alpha_idf_max"}
    missing = expected_keys - set(ckpt.keys())
    if missing:
        _fail(f"Missing checkpoint keys: {missing}")
        return None
    _pass(f"All expected keys present")

    if ckpt["dim"] != dim:
        _fail(f"Dimension mismatch: checkpoint has dim={ckpt['dim']}, encoder has dim={dim}")
        return None
    _pass(f"Dimension matches: {dim}")

    head = ImportanceHead(dim=dim)
    head.load_state_dict(ckpt["head_state_dict"])
    _pass("ImportanceHead loaded successfully")

    print(f"    alpha_idf_max: {ckpt['alpha_idf_max']:.4f}")
    if "epoch" in ckpt:
        print(f"    trained for: {ckpt['epoch']} epochs")
    if "val_mse_loss" in ckpt:
        print(f"    val MSE loss: {ckpt['val_mse_loss']:.6f}")

    return ckpt


def check_output_range(predictions: list[np.ndarray]) -> None:
    """Check 2: all predictions in [0, 1]."""
    _header("Check 2 — Output range")

    all_scores = np.concatenate(predictions)
    n_total = len(all_scores)
    n_below = int((all_scores < 0).sum())
    n_above = int((all_scores > 1).sum())
    n_nan = int(np.isnan(all_scores).sum())

    if n_nan > 0:
        _fail(f"{n_nan}/{n_total} predictions are NaN")
    else:
        _pass("No NaN predictions")

    if n_below > 0 or n_above > 0:
        _fail(f"{n_below} below 0, {n_above} above 1 out of {n_total}")
    else:
        _pass(f"All {n_total:,} predictions in [0, 1]")

    print(f"    Range: [{all_scores.min():.6f}, {all_scores.max():.6f}]")
    print(f"    Mean:  {all_scores.mean():.6f}")
    print(f"    Std:   {all_scores.std():.6f}")


def check_target_correlation(
    predictions: list[np.ndarray],
    item_ids: list[list[int]],
    alpha_idf_norm: np.ndarray,
) -> None:
    """Check 3: correlation between predicted scores and alpha_idf targets."""
    _header("Check 3 — Target correlation")

    pred_flat = []
    target_flat = []
    for preds, items in zip(predictions, item_ids, strict=True):
        for score, iid in zip(preds, items, strict=True):
            pred_flat.append(float(score))
            target_flat.append(float(alpha_idf_norm[iid]))

    pred_arr = np.array(pred_flat)
    target_arr = np.array(target_flat)

    pearson_r, pearson_p = stats.pearsonr(pred_arr, target_arr)
    spearman_r, spearman_p = stats.spearmanr(pred_arr, target_arr)

    print(f"    Pearson r:  {pearson_r:+.4f}  (p={pearson_p:.2e})")
    print(f"    Spearman ρ: {spearman_r:+.4f}  (p={spearman_p:.2e})")

    if pearson_r > 0.7:
        _pass(f"Strong Pearson correlation ({pearson_r:.4f} > 0.7)")
    elif pearson_r > 0.4:
        _warn(f"Moderate Pearson correlation ({pearson_r:.4f}) — head may need more training")
    else:
        _fail(f"Weak Pearson correlation ({pearson_r:.4f} < 0.4)")

    if spearman_r > 0.6:
        _pass(f"Good Spearman ranking correlation ({spearman_r:.4f} > 0.6)")
    elif spearman_r > 0.3:
        _warn(f"Moderate Spearman ranking ({spearman_r:.4f})")
    else:
        _fail(f"Weak Spearman ranking ({spearman_r:.4f} < 0.3)")

    # Per-item MSE
    mse = float(np.mean((pred_arr - target_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - target_arr)))
    print(f"    MSE: {mse:.6f}")
    print(f"    MAE: {mae:.6f}")


def check_per_item_agreement(
    predictions: list[np.ndarray],
    item_ids: list[list[int]],
    alpha_idf_norm: np.ndarray,
    name_map: dict[int, str],
    top_k: int = 15,
) -> None:
    """Check 4: average predicted score vs target for top/bottom items."""
    _header(f"Check 4 — Per-item predicted vs target (top/bottom {top_k})")

    # Accumulate per-item predictions
    item_pred_sum: dict[int, float] = {}
    item_pred_count: dict[int, int] = {}

    for preds, items in zip(predictions, item_ids, strict=True):
        for score, iid in zip(preds, items, strict=True):
            item_pred_sum[iid] = item_pred_sum.get(iid, 0.0) + float(score)
            item_pred_count[iid] = item_pred_count.get(iid, 0) + 1

    item_avg_pred = {
        iid: item_pred_sum[iid] / item_pred_count[iid] for iid in item_pred_sum
    }

    # Sort by target alpha_idf
    scored_items = sorted(item_avg_pred.keys(), key=lambda x: alpha_idf_norm[x], reverse=True)

    # Top items by target
    top_items = scored_items[:top_k]
    print(f"\n  {BOLD}Top {top_k} items by α_idf target:{RESET}")
    print(f"  {'ID':<7} {'Target':<10} {'Predicted':<10} {'Diff':<10} {'Name'}")
    print(f"  {'─' * 70}")
    for iid in top_items:
        target = alpha_idf_norm[iid]
        pred = item_avg_pred[iid]
        name = name_map.get(iid, "—")
        diff = pred - target
        print(f"  {iid:<7} {target:<10.4f} {pred:<10.4f} {diff:<+10.4f} {name}")

    # Bottom non-zero items by target
    nonzero_items = [iid for iid in scored_items if alpha_idf_norm[iid] > 0]
    bottom_items = nonzero_items[-top_k:]
    print(f"\n  {BOLD}Bottom {top_k} items by α_idf target:{RESET}")
    print(f"  {'ID':<7} {'Target':<10} {'Predicted':<10} {'Diff':<10} {'Name'}")
    print(f"  {'─' * 70}")
    for iid in bottom_items:
        target = alpha_idf_norm[iid]
        pred = item_avg_pred[iid]
        name = name_map.get(iid, "—")
        diff = pred - target
        print(f"  {iid:<7} {target:<10.4f} {pred:<10.4f} {diff:<+10.4f} {name}")


def check_context_sensitivity(
    predictions: list[np.ndarray],
    item_ids: list[list[int]],
) -> None:
    """Check 5: same item gets different scores in different baskets."""
    _header("Check 5 — Context sensitivity")

    item_scores: dict[int, list[float]] = {}
    for preds, items in zip(predictions, item_ids, strict=True):
        for score, iid in zip(preds, items, strict=True):
            if iid not in item_scores:
                item_scores[iid] = []
            item_scores[iid].append(float(score))

    # Only consider items with multiple appearances
    multi_items = {iid: scores for iid, scores in item_scores.items() if len(scores) >= 5}

    if not multi_items:
        _warn("Not enough items with multiple appearances to check context sensitivity")
        return

    stds = [float(np.std(scores)) for scores in multi_items.values()]
    mean_std = float(np.mean(stds))
    median_std = float(np.median(stds))
    max_std = float(np.max(stds))

    print(f"    Items with ≥5 appearances: {len(multi_items):,}")
    print(f"    Mean per-item std:   {mean_std:.6f}")
    print(f"    Median per-item std: {median_std:.6f}")
    print(f"    Max per-item std:    {max_std:.6f}")

    if mean_std > 0.001:
        _pass(
            f"Head shows context sensitivity (mean std={mean_std:.6f} > 0.001) — "
            f"same item gets different scores in different baskets"
        )
    else:
        _warn(
            f"Very low context sensitivity (mean std={mean_std:.6f}) — "
            f"head may be collapsing to global scores"
        )

    # Show example: item with highest variance
    most_variable = max(multi_items, key=lambda x: np.std(multi_items[x]))
    scores = multi_items[most_variable]
    print(
        f"\n    Most variable item (ID={most_variable}): "
        f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, "
        f"range=[{np.min(scores):.4f}, {np.max(scores):.4f}], n={len(scores)}"
    )


def check_ranking_agreement(
    predictions: list[np.ndarray],
    item_ids: list[list[int]],
    alpha_idf_norm: np.ndarray,
    top_k: int = 100,
) -> None:
    """Check 6: overlap between top-K items by predicted vs target score."""
    _header(f"Check 6 — Ranking agreement (top-{top_k})")

    # Average predicted score per item
    item_pred_sum: dict[int, float] = {}
    item_pred_count: dict[int, int] = {}
    for preds, items in zip(predictions, item_ids, strict=True):
        for score, iid in zip(preds, items, strict=True):
            item_pred_sum[iid] = item_pred_sum.get(iid, 0.0) + float(score)
            item_pred_count[iid] = item_pred_count.get(iid, 0) + 1

    item_avg_pred = {
        iid: item_pred_sum[iid] / item_pred_count[iid] for iid in item_pred_sum
    }

    # Top-K by predicted
    sorted_by_pred = sorted(item_avg_pred, key=item_avg_pred.get, reverse=True)
    top_pred = set(sorted_by_pred[:top_k])

    # Top-K by target (among items that appeared)
    sorted_by_target = sorted(item_avg_pred, key=lambda x: alpha_idf_norm[x], reverse=True)
    top_target = set(sorted_by_target[:top_k])

    overlap = len(top_pred & top_target)
    pct = overlap / top_k * 100

    print(f"    Top-{top_k} overlap: {overlap}/{top_k} ({pct:.1f}%)")

    if pct > 60:
        _pass(f"Strong ranking agreement ({pct:.1f}% overlap)")
    elif pct > 30:
        _warn(f"Moderate ranking agreement ({pct:.1f}% overlap)")
    else:
        _fail(f"Weak ranking agreement ({pct:.1f}% overlap)")

    # Also check at smaller K
    for k in [10, 50]:
        if k >= top_k:
            continue
        top_p = set(sorted_by_pred[:k])
        top_t = set(sorted_by_target[:k])
        ov = len(top_p & top_t)
        print(f"    Top-{k} overlap: {ov}/{k} ({ov / k * 100:.1f}%)")


# ── main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained ImportanceHead checkpoint."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["instacart", "tafeng", "dunnhumby"],
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to importance_head_best.pt checkpoint.",
    )
    parser.add_argument(
        "--max-baskets", type=int, default=10000,
        help="Max baskets to evaluate (default: 10000).",
    )
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    ckpt_path = Path(args.checkpoint)
    device = get_device()

    print(f"{BOLD}ImportanceHead Validation — {args.dataset}{RESET}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    # Load encoder bundle
    bundle_path = processed_dir / f"bert_encoder_bundle_{args.dataset}.pt"
    if not bundle_path.exists():
        _fail(f"Encoder bundle not found: {bundle_path}")
        return
    bundle = torch.load(bundle_path, map_location="cpu")
    dim = bundle["dim"]
    num_items = bundle["num_items"]
    item_id_offset = bundle["item_id_offset"]
    vocab_size = num_items + item_id_offset

    # Check 1: checkpoint structure
    ckpt = check_checkpoint(ckpt_path, dim)
    if ckpt is None:
        print(f"\n{RED}Aborting: checkpoint checks failed.{RESET}")
        return

    # Load models
    embedding = torch.nn.Embedding(vocab_size, dim, padding_idx=bundle["pad_token_id"])
    embedding.weight.data.copy_(bundle["state_dict"]["embedding.weight"])
    embedding = embedding.to(device).eval()

    encoder = IntraBasketEncoder(dim=dim, num_heads=4, num_layers=2, dropout=0.0)
    encoder.load_state_dict(bundle["state_dict"]["encoder"])
    encoder = encoder.to(device).eval()

    head = ImportanceHead(dim=dim)
    head.load_state_dict(ckpt["head_state_dict"])
    head = head.to(device).eval()

    # Load alpha_idf targets (normalized)
    scores_data = np.load(processed_dir / "importance_scores.npz")
    alpha_idf_raw = scores_data["alpha_idf"]
    alpha_max = ckpt["alpha_idf_max"]
    alpha_idf_norm = alpha_idf_raw / max(alpha_max, 1e-8)

    # Load baskets
    df = pl.read_parquet(processed_dir / "baskets.parquet")
    train_df, _, _ = split_user_baskets(df)
    baskets = (
        train_df.group_by(["user_id", "order_idx"])
        .agg(pl.col("item_id"))
        .select("item_id")
        .to_series()
        .to_list()
    )
    if args.max_baskets is not None:
        baskets = baskets[: args.max_baskets]

    print(f"Evaluating on {len(baskets):,} baskets...")

    # Run inference
    predictions, item_ids = _run_inference(
        embedding, encoder, head, baskets, item_id_offset, device,
    )

    # Check 2: output range
    check_output_range(predictions)

    # Check 3: target correlation
    check_target_correlation(predictions, item_ids, alpha_idf_norm)

    # Check 4: per-item agreement
    name_map = _build_name_map(processed_dir, args.dataset)
    check_per_item_agreement(predictions, item_ids, alpha_idf_norm, name_map, args.top_k)

    # Check 5: context sensitivity
    check_context_sensitivity(predictions, item_ids)

    # Check 6: ranking agreement
    check_ranking_agreement(predictions, item_ids, alpha_idf_norm)

    print(f"\n{BOLD}{'─' * 60}")
    print(f"  Validation complete.")
    print(f"{'─' * 60}{RESET}")


if __name__ == "__main__":
    main()
