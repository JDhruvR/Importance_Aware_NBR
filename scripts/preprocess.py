"""Preprocess basket datasets into a common parquet format.

Each dataset produces:
  - data/processed/{dataset}/baskets.parquet  (user_id: i32, order_idx: i32, item_id: i32)
  - data/processed/{dataset}/user2id.json
  - data/processed/{dataset}/item2id.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------


def _preprocess_basket_df(
    df: pl.DataFrame,
    min_baskets: int = 3,
    min_item_freq: int = 5,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, int]]:
    """Filter, remap, and return a standard basket DataFrame.

    Args:
        df: DataFrame with columns [user_id: str, order_idx: int, item_id: str].
            order_idx is already a per-user sequential index (0, 1, 2, …).
        min_baskets: minimum number of baskets a user must have.
        min_item_freq: minimum global frequency for an item to be kept.

    Returns:
        (remapped_df, user2id, item2id) where remapped_df has columns
        [user_id: i32, order_idx: i32, item_id: i32].
    """
    # Filter users with fewer than min_baskets
    user_counts = df.group_by("user_id").agg(pl.len().alias("n_baskets"))
    valid_users = user_counts.filter(pl.col("n_baskets") >= min_baskets)["user_id"]
    df = df.filter(pl.col("user_id").is_in(valid_users))

    # Filter items appearing fewer than min_item_freq times
    item_counts = df.group_by("item_id").agg(pl.len().alias("freq"))
    valid_items = item_counts.filter(pl.col("freq") >= min_item_freq)["item_id"]
    df = df.filter(pl.col("item_id").is_in(valid_items))

    # Remap user IDs to contiguous integers
    unique_users = sorted(df["user_id"].unique().to_list())
    user2id = {u: i for i, u in enumerate(unique_users)}

    # Remap item IDs to contiguous integers
    unique_items = sorted(df["item_id"].unique().to_list())
    item2id = {it: i for i, it in enumerate(unique_items)}

    remapped = df.with_columns(
        pl.col("user_id").replace_strict(user2id).cast(pl.Int32),
        pl.col("item_id").replace_strict(item2id).cast(pl.Int32),
        pl.col("order_idx").cast(pl.Int32),
    ).select(["user_id", "order_idx", "item_id"])

    return remapped, user2id, item2id


# ---------------------------------------------------------------------------
# Instacart
# ---------------------------------------------------------------------------


def preprocess_instacart(
    raw_dir: Path,
    processed_dir: Path,
    min_baskets: int = 3,
    min_item_freq: int = 5,
) -> None:
    """Preprocess Instacart 2017 dataset."""
    orders = pl.read_csv(raw_dir / "orders.csv")
    prior = pl.read_csv(raw_dir / "order_products__prior.csv")
    train = pl.read_csv(raw_dir / "order_products__train.csv")

    # Merge prior and train product splits
    products = pl.concat([prior, train], how="vertical")

    # Merge with orders to get user_id and order_number
    merged = orders.join(products, on="order_id", how="inner")

    # Build (user_id, order_sequence_index, item_id) long table
    # order_number is already per-user sequential (1, 2, 3, …)
    basket_df = merged.select(
        pl.col("user_id").cast(pl.Utf8),
        (pl.col("order_number") - 1).alias("order_idx"),  # 0-indexed
        pl.col("product_id").cast(pl.Utf8).alias("item_id"),
    )

    remapped, user2id, item2id = _preprocess_basket_df(
        basket_df,
        min_baskets=min_baskets,
        min_item_freq=min_item_freq,
    )

    _save(processed_dir, remapped, user2id, item2id, "instacart")


# ---------------------------------------------------------------------------
# Dunnhumby
# ---------------------------------------------------------------------------


def preprocess_dunnhumby(
    raw_dir: Path,
    processed_dir: Path,
    min_baskets: int = 3,
    min_item_freq: int = 5,
) -> None:
    """Preprocess Dunnhumby Complete Journey dataset."""
    # Sample files are split by month (transactions_YYYYMM.csv)
    transaction_files = sorted(raw_dir.glob("transactions_*.csv"))
    if not transaction_files:
        raise RuntimeError(f"No transaction files found in {raw_dir}. Expected transactions_*.csv")

    # Load and concatenate all transaction files (select only required columns)
    required = ["CUST_CODE", "BASKET_ID", "SHOP_DATE", "PROD_CODE"]
    frames: list[pl.DataFrame] = []
    for f in transaction_files:
        df = pl.read_csv(f)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(
                f"{f.name} missing columns: {missing}. Verify the extracted Dunnhumby sample."
            )
        frames.append(
            df.select(required).with_columns(
                pl.col("CUST_CODE").cast(pl.Utf8),
                pl.col("PROD_CODE").cast(pl.Utf8),
                pl.col("BASKET_ID").cast(pl.Utf8),
                pl.col("SHOP_DATE").cast(pl.Int64),
            )
        )
    transactions = pl.concat(frames, how="vertical")

    if "CUST_CODE" not in transactions.columns:
        raise RuntimeError(
            "Dunnhumby sample files missing CUST_CODE. Verify the correct dataset was extracted."
        )
    if "PROD_CODE" not in transactions.columns:
        raise RuntimeError(
            "Dunnhumby sample files missing PROD_CODE. Verify the correct dataset was extracted."
        )
    if "BASKET_ID" not in transactions.columns or "SHOP_DATE" not in transactions.columns:
        raise RuntimeError(
            "Dunnhumby sample files missing BASKET_ID/SHOP_DATE. Verify the correct dataset was extracted."
        )

    # Columns (sample dataset): SHOP_DATE, CUST_CODE, BASKET_ID, PROD_CODE
    # SHOP_DATE is YYYYMMDD; BASKET_ID is unique per shopping trip.
    # Build a sequential order_idx per customer based on SHOP_DATE and BASKET_ID.
    basket_days = (
        transactions.select("CUST_CODE", "BASKET_ID", "SHOP_DATE")
        .unique()
        .sort(["CUST_CODE", "SHOP_DATE", "BASKET_ID"])
    )

    basket_days = basket_days.with_columns(
        pl.col("BASKET_ID").rank(method="ordinal").over("CUST_CODE").alias("order_idx") - 1,
    )

    merged = transactions.join(
        basket_days.select("CUST_CODE", "BASKET_ID", "order_idx"),
        on=["CUST_CODE", "BASKET_ID"],
        how="inner",
    )

    basket_df = merged.select(
        pl.col("CUST_CODE").cast(pl.Utf8).alias("user_id"),
        pl.col("order_idx"),
        pl.col("PROD_CODE").cast(pl.Utf8).alias("item_id"),
    )

    remapped, user2id, item2id = _preprocess_basket_df(
        basket_df,
        min_baskets=min_baskets,
        min_item_freq=min_item_freq,
    )

    _save(processed_dir, remapped, user2id, item2id, "dunnhumby")


# ---------------------------------------------------------------------------
# TaFeng
# ---------------------------------------------------------------------------


def preprocess_tafeng(
    raw_dir: Path,
    processed_dir: Path,
    min_baskets: int = 3,
    min_item_freq: int = 5,
) -> None:
    """Preprocess TaFeng grocery dataset."""
    # The merged Kaggle file is ta_feng_all_months_merged.csv
    df = pl.read_csv(raw_dir / "ta_feng_all_months_merged.csv")

    # Columns: TRANSACTION_DT, CUSTOMER_ID, PRODUCT_ID, …
    # Parse dates and sort to establish temporal order
    df = df.with_columns(
        pl.col("TRANSACTION_DT").str.to_date("%m/%d/%Y").alias("date"),
    )

    # Create a basket key from (customer, date) — same-day purchases = same basket
    basket_keys = (
        df.select("CUSTOMER_ID", "date")
        .unique()
        .sort(["CUSTOMER_ID", "date"])
        .with_columns(
            pl.col("CUSTOMER_ID").rank(method="ordinal").over("CUSTOMER_ID").alias("order_idx") - 1,
        )
    )

    # Join back
    merged = df.join(
        basket_keys.select("CUSTOMER_ID", "date", "order_idx"),
        on=["CUSTOMER_ID", "date"],
        how="inner",
    )

    basket_df = merged.select(
        pl.col("CUSTOMER_ID").cast(pl.Utf8).alias("user_id"),
        pl.col("order_idx"),
        pl.col("PRODUCT_ID").cast(pl.Utf8).alias("item_id"),
    )

    remapped, user2id, item2id = _preprocess_basket_df(
        basket_df,
        min_baskets=min_baskets,
        min_item_freq=min_item_freq,
    )

    _save(processed_dir, remapped, user2id, item2id, "tafeng")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(
    processed_dir: Path,
    df: pl.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    dataset_name: str,
) -> None:
    """Save parquet and ID maps."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(processed_dir / "baskets.parquet")

    with open(processed_dir / "user2id.json", "w") as f:
        json.dump(user2id, f)
    with open(processed_dir / "item2id.json", "w") as f:
        json.dump(item2id, f)

    n_users = len(user2id)
    n_items = len(item2id)
    n_rows = len(df)
    print(f"[{dataset_name}] Saved {n_rows:,} rows | {n_users:,} users | {n_items:,} items")
    print(f"  Parquet: {processed_dir / 'baskets.parquet'}")
    print(f"  user2id: {processed_dir / 'user2id.json'}")
    print(f"  item2id: {processed_dir / 'item2id.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "instacart": preprocess_instacart,
    "dunnhumby": preprocess_dunnhumby,
    "tafeng": preprocess_tafeng,
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_root = root / "data" / "raw"
    processed_root = root / "data" / "processed"

    if len(sys.argv) < 2:
        print("Usage: python scripts/preprocess.py <dataset> [dataset ...]")
        print(f"Available: {', '.join(DATASETS)}")
        sys.exit(1)

    for name in sys.argv[1:]:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {', '.join(DATASETS)}")
            sys.exit(1)

        raw_dir = raw_root / name
        if not raw_dir.exists():
            print(f"[{name}] Raw directory not found: {raw_dir}")
            print("Run: python scripts/download_data.py first.")
            sys.exit(1)

        processed_dir = processed_root / name
        DATASETS[name](raw_dir, processed_dir)


if __name__ == "__main__":
    main()
