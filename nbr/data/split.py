"""Train / val / test split for basket sequences.

Split rule: for each user, last order -> test, second-to-last -> val,
remainder -> train. Users with exactly 3 orders get 1 train, 1 val, 1 test basket.
"""

from __future__ import annotations

import polars as pl


def split_user_baskets(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split basket DataFrame into train, val, test per user.

    Args:
        df: DataFrame with columns [user_id: i32, order_idx: i32, item_id: i32].

    Returns:
        (train_df, val_df, test_df) — each with the same schema as input.
    """
    # Find max order_idx per user (their last basket index)
    max_order = df.group_by("user_id").agg(
        pl.col("order_idx").max().alias("max_order"),
    )

    # Tag each row: test if order_idx == max_order, val if == max_order - 1, else train
    df = df.join(max_order, on="user_id", how="left")
    df = df.with_columns(
        pl.when(pl.col("order_idx") == pl.col("max_order"))
        .then(pl.lit("test"))
        .when(pl.col("order_idx") == pl.col("max_order") - 1)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split"),
    ).drop("max_order")

    train_df = df.filter(pl.col("split") == "train").drop("split")
    val_df = df.filter(pl.col("split") == "val").drop("split")
    test_df = df.filter(pl.col("split") == "test").drop("split")

    # Sanity check: union equals original
    assert len(train_df) + len(val_df) + len(test_df) == len(df), (
        f"Split sizes don't sum to total: {len(train_df)} + {len(val_df)} + {len(test_df)} != {len(df)}"
    )

    return train_df, val_df, test_df
