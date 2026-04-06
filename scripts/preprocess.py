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


def _build_maps(
    user2id: dict[str, int],
    item2id: dict[str, int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build mapping DataFrames for joins."""
    user_map = pl.DataFrame(
        {"user_id_raw": list(user2id.keys()), "user_id": list(user2id.values())}
    )
    item_map = pl.DataFrame(
        {"item_id_raw": list(item2id.keys()), "item_id": list(item2id.values())}
    )
    return user_map, item_map


def _map_user(df: pl.DataFrame, user_map: pl.DataFrame, user_col: str) -> pl.DataFrame:
    """Map raw user IDs to contiguous user_id."""
    return (
        df.with_columns(pl.col(user_col).cast(pl.Utf8).alias("user_id_raw"))
        .drop(user_col)
        .join(user_map, on="user_id_raw", how="inner")
        .drop("user_id_raw")
    )


def _map_item(df: pl.DataFrame, item_map: pl.DataFrame, item_col: str) -> pl.DataFrame:
    """Map raw item IDs to contiguous item_id."""
    return (
        df.with_columns(pl.col(item_col).cast(pl.Utf8).alias("item_id_raw"))
        .drop(item_col)
        .join(item_map, on="item_id_raw", how="inner")
        .drop("item_id_raw")
    )


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
    products = pl.read_csv(raw_dir / "products.csv")
    aisles = pl.read_csv(raw_dir / "aisles.csv")
    departments = pl.read_csv(raw_dir / "departments.csv")

    # Merge prior and train product splits
    products_all = pl.concat([prior, train], how="vertical")

    # Merge with orders to get user_id and order_number
    merged = orders.join(products_all, on="order_id", how="inner")

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

    user_map, item_map = _build_maps(user2id, item2id)

    # Item metadata table
    items_meta = (
        products.join(aisles, on="aisle_id", how="left")
        .join(departments, on="department_id", how="left")
        .select(
            pl.col("product_id").cast(pl.Utf8).alias("item_id_raw"),
            pl.col("product_id").alias("product_id_raw"),
            pl.col("product_name"),
            pl.col("aisle_id").cast(pl.Int32),
            pl.col("aisle"),
            pl.col("department_id").cast(pl.Int32),
            pl.col("department"),
        )
        .join(item_map, on="item_id_raw", how="inner")
        .drop("item_id_raw")
        .select(
            "item_id",
            "product_id_raw",
            "product_name",
            "aisle_id",
            "aisle",
            "department_id",
            "department",
        )
    )
    items_meta = _cast_i32(items_meta, ["item_id", "aisle_id", "department_id"])
    items_meta = _cast_i32(items_meta, ["item_id", "product_subclass"])
    items_meta = _cast_i32(items_meta, ["item_id"])
    items_meta = items_meta.with_columns(pl.col("item_id").cast(pl.Int32))
    items_meta.write_parquet(processed_dir / "items.parquet")

    # Basket metadata
    basket_meta = orders.select(
        pl.col("user_id").cast(pl.Utf8),
        (pl.col("order_number") - 1).alias("order_idx"),
        pl.col("order_dow").cast(pl.Int16),
        pl.col("order_hour_of_day").cast(pl.Int16),
        pl.col("days_since_prior_order").cast(pl.Float32),
    )
    basket_meta = _map_user(basket_meta, user_map, "user_id")
    basket_meta = _cast_i32(basket_meta, ["user_id", "order_idx", "order_dow", "order_hour_of_day"])
    basket_meta.write_parquet(processed_dir / "basket_meta.parquet")

    # Basket-item metadata
    basket_items = merged.select(
        pl.col("user_id").cast(pl.Utf8),
        (pl.col("order_number") - 1).alias("order_idx"),
        pl.col("product_id").cast(pl.Utf8).alias("item_id_raw"),
        pl.col("add_to_cart_order").cast(pl.Int16),
        pl.col("reordered").cast(pl.Int8),
    )
    basket_items = _map_user(basket_items, user_map, "user_id")
    basket_items = basket_items.join(item_map, on="item_id_raw", how="inner").drop("item_id_raw")
    basket_items = _cast_i32(
        basket_items, ["user_id", "order_idx", "item_id", "add_to_cart_order", "reordered"]
    )
    basket_items.write_parquet(processed_dir / "basket_items.parquet")


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
        # Add optional product hierarchy columns if absent
        for col in ["PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(
                f"{f.name} missing columns: {missing}. Verify the extracted Dunnhumby sample."
            )
        cleaned = (
            df.select(required + ["PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"])
            .with_columns(
                pl.col("CUST_CODE").cast(pl.Utf8),
                pl.col("PROD_CODE").cast(pl.Utf8),
                pl.col("PROD_CODE_10").cast(pl.Utf8),
                pl.col("PROD_CODE_20").cast(pl.Utf8),
                pl.col("PROD_CODE_30").cast(pl.Utf8),
                pl.col("PROD_CODE_40").cast(pl.Utf8),
                pl.col("BASKET_ID").cast(pl.Utf8),
                pl.col("SHOP_DATE").cast(pl.Int64),
            )
            .filter(pl.col("CUST_CODE").is_not_null() & (pl.col("CUST_CODE") != ""))
        )
        frames.append(cleaned)
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
        pl.int_range(0, pl.len()).over("CUST_CODE").alias("order_idx"),
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

    user_map, item_map = _build_maps(user2id, item2id)

    # Item metadata table
    items_meta = (
        transactions.select(
            pl.col("PROD_CODE").cast(pl.Utf8).alias("item_id_raw"),
            pl.col("PROD_CODE").cast(pl.Utf8).alias("prod_code_raw"),
            pl.col("PROD_CODE_10").cast(pl.Utf8).alias("prod_code_10"),
            pl.col("PROD_CODE_20").cast(pl.Utf8).alias("prod_code_20"),
            pl.col("PROD_CODE_30").cast(pl.Utf8).alias("prod_code_30"),
            pl.col("PROD_CODE_40").cast(pl.Utf8).alias("prod_code_40"),
        )
        .unique()
        .join(item_map, on="item_id_raw", how="inner")
        .drop("item_id_raw")
        .select(
            "item_id",
            "prod_code_raw",
            "prod_code_10",
            "prod_code_20",
            "prod_code_30",
            "prod_code_40",
        )
    )
    items_meta = items_meta.with_columns(pl.col("item_id").cast(pl.Int32))
    items_meta.write_parquet(processed_dir / "items.parquet")

    # Basket metadata (sample only includes SHOP_DATE)
    basket_meta = (
        transactions.select(
            pl.col("CUST_CODE").cast(pl.Utf8),
            pl.col("BASKET_ID").cast(pl.Utf8),
            pl.col("SHOP_DATE").cast(pl.Int64),
        )
        .unique()
        .join(
            basket_days.select("CUST_CODE", "BASKET_ID", "order_idx"),
            on=["CUST_CODE", "BASKET_ID"],
            how="inner",
        )
        .drop("BASKET_ID")
    )
    basket_meta = _map_user(basket_meta, user_map, "CUST_CODE")
    basket_meta = _cast_i32(basket_meta, ["user_id", "order_idx"])
    basket_meta.write_parquet(processed_dir / "basket_meta.parquet")

    # Basket-item metadata (sample lacks quantity/spend)
    basket_items = transactions.select(
        pl.col("CUST_CODE").cast(pl.Utf8),
        pl.col("BASKET_ID").cast(pl.Utf8),
        pl.col("PROD_CODE").cast(pl.Utf8).alias("item_id_raw"),
    )
    basket_items = basket_items.join(
        basket_days.select("CUST_CODE", "BASKET_ID", "order_idx"),
        on=["CUST_CODE", "BASKET_ID"],
        how="inner",
    ).drop("BASKET_ID")
    basket_items = _map_user(basket_items, user_map, "CUST_CODE")
    basket_items = basket_items.join(item_map, on="item_id_raw", how="inner").drop("item_id_raw")
    basket_items = _cast_i32(basket_items, ["user_id", "order_idx", "item_id"])
    basket_items.write_parquet(processed_dir / "basket_items.parquet")


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
            pl.int_range(0, pl.len()).over("CUSTOMER_ID").alias("order_idx"),
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

    user_map, item_map = _build_maps(user2id, item2id)

    # Item metadata table
    items_meta = (
        df.select(
            pl.col("PRODUCT_ID").cast(pl.Utf8).alias("item_id_raw"),
            pl.col("PRODUCT_ID").cast(pl.Utf8).alias("product_id_raw"),
            pl.col("PRODUCT_SUBCLASS").cast(pl.Int32).alias("product_subclass"),
        )
        .unique()
        .join(item_map, on="item_id_raw", how="inner")
        .drop("item_id_raw")
        .select("item_id", "product_id_raw", "product_subclass")
        .with_columns(pl.col("item_id").cast(pl.Int32))
    )
    items_meta.write_parquet(processed_dir / "items.parquet")

    # User metadata table
    user_meta = df.select(
        pl.col("CUSTOMER_ID").cast(pl.Utf8),
        pl.col("AGE_GROUP").cast(pl.Utf8),
        pl.col("PIN_CODE").cast(pl.Utf8),
    ).unique()
    user_meta = _map_user(user_meta, user_map, "CUSTOMER_ID")
    user_meta = _cast_i32(user_meta, ["user_id"])
    user_meta.write_parquet(processed_dir / "user_meta.parquet")

    # Basket metadata (transaction date)
    basket_meta = basket_keys.select(
        pl.col("CUSTOMER_ID").cast(pl.Utf8),
        pl.col("order_idx").cast(pl.Int32),
        pl.col("date"),
    )
    basket_meta = _map_user(basket_meta, user_map, "CUSTOMER_ID")
    basket_meta = _cast_i32(basket_meta, ["user_id", "order_idx"])
    basket_meta.write_parquet(processed_dir / "basket_meta.parquet")

    # Basket-item metadata
    basket_items = merged.select(
        pl.col("CUSTOMER_ID").cast(pl.Utf8),
        pl.col("order_idx").cast(pl.Int32),
        pl.col("PRODUCT_ID").cast(pl.Utf8).alias("item_id_raw"),
        pl.col("AMOUNT").cast(pl.Float32),
        pl.col("ASSET").cast(pl.Float32),
        pl.col("SALES_PRICE").cast(pl.Float32),
    )
    basket_items = _map_user(basket_items, user_map, "CUSTOMER_ID")
    basket_items = basket_items.join(item_map, on="item_id_raw", how="inner").drop("item_id_raw")
    basket_items = _cast_i32(basket_items, ["user_id", "order_idx", "item_id"])
    basket_items.write_parquet(processed_dir / "basket_items.parquet")


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


def _cast_i32(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Cast specified columns to int32 if present."""
    for col in cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
    return df


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
