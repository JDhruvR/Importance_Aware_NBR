"""Validate importance scores produced by compute_importance.py.

Checks performed:
  1. File existence and array shapes / dtypes
  2. Value ranges (no NaN, Inf, negatives; raw_importance in [0,1])
  3. Consistency with baskets.parquet (coverage, alpha ≈ raw × idf)
  4. Distribution statistics and percentiles
  5. Top / bottom items with product names
  6. Correlation between raw_importance and idf_factor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

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


# ── item name lookup (mirrors check_embeddings.py) ──────────────────
def _build_name_map(processed_dir: Path, dataset: str) -> dict[int, str]:
    """Build item_id → human-readable name mapping."""
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


# ── checks ───────────────────────────────────────────────────────────
def check_file_and_arrays(npz_path: Path) -> dict[str, np.ndarray] | None:
    """Check 1: file exists, expected keys present, shapes and dtypes ok."""
    _header("Check 1 — File structure")

    if not npz_path.exists():
        _fail(f"File not found: {npz_path}")
        return None
    _pass(f"File exists: {npz_path}")

    data = dict(np.load(npz_path))
    expected_keys = {"alpha_idf", "raw_importance", "idf_factor"}
    missing = expected_keys - set(data.keys())
    extra = set(data.keys()) - expected_keys
    if missing:
        _fail(f"Missing keys: {missing}")
        return None
    _pass(f"All expected keys present: {sorted(data.keys())}")
    if extra:
        _warn(f"Unexpected extra keys: {extra}")

    shapes = {k: v.shape for k, v in data.items()}
    ref_shape = data["alpha_idf"].shape
    if all(s == ref_shape for s in shapes.values()):
        _pass(f"All arrays have consistent shape: {ref_shape}")
    else:
        _fail(f"Shape mismatch: {shapes}")
        return None

    if len(ref_shape) != 1:
        _fail(f"Expected 1-D arrays, got shape {ref_shape}")
        return None
    _pass(f"Arrays are 1-D with {ref_shape[0]} items")

    for k, v in data.items():
        if np.issubdtype(v.dtype, np.floating):
            _pass(f"  {k}: dtype={v.dtype}")
        else:
            _warn(f"  {k}: unexpected dtype={v.dtype}")

    return data


def check_value_ranges(data: dict[str, np.ndarray]) -> None:
    """Check 2: no NaN/Inf, no negatives, raw_importance in [0,1]."""
    _header("Check 2 — Value ranges")
    all_ok = True

    for key in ["alpha_idf", "raw_importance", "idf_factor"]:
        arr = data[key]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        n_neg = int((arr < 0).sum())

        if n_nan > 0:
            _fail(f"{key}: {n_nan} NaN values")
            all_ok = False
        if n_inf > 0:
            _fail(f"{key}: {n_inf} Inf values")
            all_ok = False
        if n_neg > 0:
            _fail(f"{key}: {n_neg} negative values")
            all_ok = False

    if all_ok:
        _pass("No NaN, Inf, or negative values in any array")

    raw = data["raw_importance"]
    valid_raw = raw[raw > 0]
    if len(valid_raw) > 0 and valid_raw.max() <= 1.0:
        _pass(f"raw_importance max = {valid_raw.max():.6f} (≤ 1.0 as expected)")
    elif len(valid_raw) > 0:
        _warn(f"raw_importance max = {valid_raw.max():.6f} (> 1.0, check normalization)")


def check_consistency(data: dict[str, np.ndarray], processed_dir: Path) -> None:
    """Check 3: cross-reference with baskets.parquet."""
    _header("Check 3 — Consistency with dataset")

    baskets_path = processed_dir / "baskets.parquet"
    if not baskets_path.exists():
        _warn(f"baskets.parquet not found at {baskets_path}, skipping consistency checks")
        return

    df = pl.read_parquet(baskets_path)
    num_items_in_baskets = df["item_id"].n_unique()
    num_items_in_scores = len(data["alpha_idf"])

    if num_items_in_scores >= num_items_in_baskets:
        _pass(
            f"Score array covers all items: {num_items_in_scores} slots "
            f"≥ {num_items_in_baskets} unique items in baskets"
        )
    else:
        _fail(
            f"Score array too small: {num_items_in_scores} < {num_items_in_baskets} unique items"
        )

    # Check coverage: how many items have non-zero scores
    n_nonzero = int((data["alpha_idf"] > 0).sum())
    coverage = n_nonzero / max(num_items_in_baskets, 1) * 100
    if coverage > 90:
        _pass(f"Coverage: {n_nonzero}/{num_items_in_baskets} items have non-zero α_idf ({coverage:.1f}%)")
    elif coverage > 50:
        _warn(f"Coverage: {n_nonzero}/{num_items_in_baskets} items ({coverage:.1f}%) — some items may only appear in val/test")
    else:
        _fail(f"Coverage: {n_nonzero}/{num_items_in_baskets} items ({coverage:.1f}%) — too low")

    # Check multiplicative consistency: alpha_idf ≈ raw_importance × idf_factor
    alpha = data["alpha_idf"]
    raw = data["raw_importance"]
    idf = data["idf_factor"]
    recomputed = raw * idf

    valid = alpha > 0
    if valid.sum() > 0:
        max_diff = float(np.max(np.abs(alpha[valid] - recomputed[valid])))
        if max_diff < 1e-4:
            _pass(f"α_idf ≈ raw × idf (max absolute diff = {max_diff:.2e})")
        else:
            _fail(f"α_idf ≠ raw × idf (max absolute diff = {max_diff:.2e})")

    # Check that zero patterns are consistent
    zero_raw = raw == 0
    zero_idf = idf == 0
    zero_alpha = alpha == 0
    if np.all(zero_raw == zero_alpha) or np.all(zero_idf == zero_alpha):
        _pass("Zero patterns are consistent across arrays")
    else:
        _warn("Some items have zero in one array but non-zero in another")


def check_distributions(data: dict[str, np.ndarray]) -> None:
    """Check 4: print distribution statistics."""
    _header("Check 4 — Distribution statistics")

    for key in ["alpha_idf", "raw_importance", "idf_factor"]:
        arr = data[key]
        valid = arr[arr > 0]
        if len(valid) == 0:
            _fail(f"{key}: no non-zero values!")
            continue

        pcts = np.percentile(valid, [5, 25, 50, 75, 95])
        print(f"\n  {BOLD}{key}{RESET} (n={len(valid)} non-zero / {len(arr)} total)")
        print(f"    Mean:  {valid.mean():.6f}")
        print(f"    Std:   {valid.std():.6f}")
        print(f"    Min:   {valid.min():.6f}")
        print(f"     5th:  {pcts[0]:.6f}")
        print(f"    25th:  {pcts[1]:.6f}")
        print(f"    50th:  {pcts[2]:.6f}")
        print(f"    75th:  {pcts[3]:.6f}")
        print(f"    95th:  {pcts[4]:.6f}")
        print(f"    Max:   {valid.max():.6f}")

    # Skewness check for alpha_idf
    valid_alpha = data["alpha_idf"][data["alpha_idf"] > 0]
    if len(valid_alpha) > 0:
        mean_val = valid_alpha.mean()
        median_val = float(np.median(valid_alpha))
        if mean_val > median_val:
            _pass(f"α_idf is right-skewed (mean={mean_val:.4f} > median={median_val:.4f}) — expected")
        else:
            _warn(f"α_idf is not right-skewed (mean={mean_val:.4f} ≤ median={median_val:.4f})")


def check_top_bottom_items(
    data: dict[str, np.ndarray], processed_dir: Path, dataset: str, top_k: int = 20
) -> None:
    """Check 5: show top and bottom items with product names."""
    _header(f"Check 5 — Top {top_k} and bottom 10 items")

    name_map = _build_name_map(processed_dir, dataset)
    if not name_map:
        _warn("Could not load item names; showing IDs only")

    alpha = data["alpha_idf"]
    raw = data["raw_importance"]
    idf = data["idf_factor"]

    # Top items
    top_idx = np.argsort(alpha)[-top_k:][::-1]
    print(f"\n  {BOLD}Top {top_k} items by α_idf:{RESET}")
    print(f"  {'Rank':<5} {'ID':<7} {'α_idf':<10} {'raw_Δ':<10} {'IDF':<8} {'Name'}")
    print(f"  {'─' * 75}")
    for rank, idx in enumerate(top_idx, 1):
        name = name_map.get(int(idx), "—")
        print(f"  {rank:<5} {idx:<7} {alpha[idx]:<10.4f} {raw[idx]:<10.4f} {idf[idx]:<8.2f} {name}")

    # Bottom non-zero items
    nonzero_mask = alpha > 0
    if nonzero_mask.sum() > 0:
        nonzero_indices = np.where(nonzero_mask)[0]
        sorted_nonzero = nonzero_indices[np.argsort(alpha[nonzero_indices])]
        bot_idx = sorted_nonzero[:10]

        print(f"\n  {BOLD}Bottom 10 items by α_idf (non-zero):{RESET}")
        print(f"  {'Rank':<5} {'ID':<7} {'α_idf':<10} {'raw_Δ':<10} {'IDF':<8} {'Name'}")
        print(f"  {'─' * 75}")
        for rank, idx in enumerate(bot_idx, 1):
            name = name_map.get(int(idx), "—")
            print(
                f"  {rank:<5} {idx:<7} {alpha[idx]:<10.4f} {raw[idx]:<10.4f} {idf[idx]:<8.2f} {name}"
            )


def check_correlations(data: dict[str, np.ndarray]) -> None:
    """Check 6: correlations between components."""
    _header("Check 6 — Component correlations")

    valid = data["alpha_idf"] > 0
    if valid.sum() < 10:
        _warn("Too few non-zero items to compute correlations")
        return

    raw = data["raw_importance"][valid]
    idf = data["idf_factor"][valid]
    alpha = data["alpha_idf"][valid]

    corr_raw_idf = float(np.corrcoef(raw, idf)[0, 1])
    corr_raw_alpha = float(np.corrcoef(raw, alpha)[0, 1])
    corr_idf_alpha = float(np.corrcoef(idf, alpha)[0, 1])

    print(f"  Pearson correlations (non-zero items only, n={int(valid.sum())}):")
    print(f"    raw_importance ↔ idf_factor : {corr_raw_idf:+.4f}")
    print(f"    raw_importance ↔ alpha_idf  : {corr_raw_alpha:+.4f}")
    print(f"    idf_factor     ↔ alpha_idf  : {corr_idf_alpha:+.4f}")

    # Sanity: both components should contribute to alpha
    if abs(corr_raw_alpha) > 0.3 and abs(corr_idf_alpha) > 0.3:
        _pass("Both raw_importance and idf_factor contribute meaningfully to α_idf")
    elif abs(corr_raw_alpha) < 0.1:
        _warn("raw_importance has very low correlation with α_idf — IDF may be dominating")
    elif abs(corr_idf_alpha) < 0.1:
        _warn("idf_factor has very low correlation with α_idf — raw deltas may be dominating")


# ── main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate importance scores from compute_importance.py."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["instacart", "tafeng", "dunnhumby"],
        help="Name of the dataset to validate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top items to display (default: 20).",
    )
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    npz_path = processed_dir / "importance_scores.npz"

    print(f"{BOLD}Importance Score Validation — {args.dataset}{RESET}")
    print(f"File: {npz_path}")

    # Check 1
    data = check_file_and_arrays(npz_path)
    if data is None:
        print(f"\n{RED}Aborting: file/array checks failed.{RESET}")
        return

    # Check 2
    check_value_ranges(data)

    # Check 3
    check_consistency(data, processed_dir)

    # Check 4
    check_distributions(data)

    # Check 5
    check_top_bottom_items(data, processed_dir, args.dataset, top_k=args.top_k)

    # Check 6
    check_correlations(data)

    print(f"\n{BOLD}{'─' * 60}")
    print(f"  Validation complete.")
    print(f"{'─' * 60}{RESET}")


if __name__ == "__main__":
    main()
