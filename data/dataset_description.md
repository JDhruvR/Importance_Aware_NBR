# Dataset Description

This document describes the three datasets we preprocess into a unified basket format
and the resulting processed schema/statistics. All processed files live under
`data/processed/{dataset}/` and share the same schema.

## Common Processed Schema

File: `data/processed/{dataset}/baskets.parquet`

Columns:
- `user_id` (int32): contiguous user index starting at 0
- `order_idx` (int32): 0-indexed basket sequence position per user
- `item_id` (int32): contiguous item index starting at 0

Baskets are stored as **item-level rows**, not as nested lists.

- Each row in `data/processed/{dataset}/baskets.parquet` is a single `(user_id, order_idx, item_id)` triple.
- A **basket** is the set of rows with the same `(user_id, order_idx)`.
- `order_idx` is a per-user sequence index (0-based).

So variable-length baskets are handled by grouping rows per basket at runtime:

- `BasketSequenceDataset` groups all items for each `(user_id, order_idx)` into a list.
- `BasketCollator` pads baskets and sequences to produce tensors:
  - `items`: (B, T, S)
  - `item_mask`: (B, T, S)
  - `basket_mask`: (B, T)

**Product features:**  
Right now **only the item ID** is used. The “feature vector” for each product is simply a **learned embedding** from `ItemEmbedding`. We do **not** include product metadata like category, aisle, price, etc. Those fields are not kept in the processed dataset at the moment.

**Notes:**

- Each row represents a single item within a basket.
- `order_idx` is constructed per user from the raw dataset’s temporal fields.
- Users with < 3 baskets and items with < 5 appearances are filtered out.

## Instacart (Kaggle)

Raw source: `data/raw/instacart/`

Raw files used:
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`

Raw key fields:
- `orders.csv`: `order_id`, `user_id`, `order_number`
- `order_products__*.csv`: `order_id`, `product_id`

Processing notes:
- Prior + train splits are merged before joining with orders.
- `order_number` is converted to `order_idx = order_number - 1`.

Processed stats:
- Rows: 33,813,577
- Users: 206,209
- Items: 47,975
- Baskets per user: min 2 / p25 6 / median 10 / mean 16.23 / p75 20 / max 100
- Items per basket: min 1 / p25 5 / median 8 / mean 10.11 / p75 14 / max 145

## TaFeng (Kaggle)

Raw source: `data/raw/tafeng/`

Raw file used:
- `ta_feng_all_months_merged.csv`

Raw key fields:
- `TRANSACTION_DT` (mm/dd/yyyy)
- `CUSTOMER_ID`
- `PRODUCT_ID`

Processing notes:
- Same-day purchases are treated as one basket per customer.
- `order_idx` is derived from sorted transaction dates per user.

Processed stats:
- Rows: 795,321
- Users: 29,197
- Items: 15,743
- Baskets per user: min 1 / p25 1 / median 2 / mean 3.97 / p75 5 / max 86
- Items per basket: min 1 / p25 2 / median 5 / mean 6.87 / p75 9 / max 111

## Dunnhumby (Sample ZIP)

Raw source: `data/raw/dunnhumby/`

Raw files used:
- `transactions_*.csv` (multiple monthly files)
- `time.csv` (not used in preprocessing)

Raw key fields:
- `CUST_CODE`
- `BASKET_ID`
- `SHOP_DATE` (yyyymmdd integer)
- `PROD_CODE`

Processing notes:
- Rows with missing `CUST_CODE` are filtered out.
- `order_idx` is a 0-indexed sequence per user based on sorted `SHOP_DATE` + `BASKET_ID`.
- Multiple transaction files are concatenated after selecting only the required columns.

Processed stats:
- Rows: 25,193,229
- Users: 48,125
- Items: 4,997
- Baskets per user: min 1 / p25 8 / median 41 / mean 78.75 / p75 115 / max 1,157
- Items per basket: min 1 / p25 2 / median 4 / mean 6.65 / p75 9 / max 101

## How to Use in Models

Typical flow:
1. Load `baskets.parquet`
2. Split into train/val/test with `split_user_baskets()`
3. Build a `BasketSequenceDataset` using the train split
4. Use `BasketCollator` to create padded batches

All models expect batch tensors with:
- `items`: (B, T, S)
- `item_mask`: (B, T, S)
- `basket_mask`: (B, T)
- `target`: (B, V)
