# Dataset Description

This document describes the three datasets we preprocess into a unified basket format
and the resulting processed schema/statistics. All processed files live under
`data/processed/{dataset}/`.

## Common Processed Tables

### 1) baskets.parquet (shared across datasets)

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

Notes:
- Each row represents a single item within a basket.
- `order_idx` is constructed per user from the raw dataset’s temporal fields.
- Users with < 3 baskets and items with < 5 appearances are filtered out.

### 2) items.parquet (item metadata)

Each dataset has an item metadata table with all available item-level attributes.

### 3) basket_meta.parquet (basket-level metadata)

Optional basket-level fields derived from raw orders/transactions.

### 4) basket_items.parquet (item-in-basket metadata)

Optional per-item purchase attributes (e.g., reorder flags, amounts).

### 5) user_meta.parquet (user metadata, if available)

Optional user-level attributes when present in raw data.

## Instacart (Kaggle)

Raw source: `data/raw/instacart/`

Raw files used:
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`, `aisles.csv`, `departments.csv`

Raw key fields:
- `orders.csv`: `order_id`, `user_id`, `order_number`
- `order_products__*.csv`: `order_id`, `product_id`

Processing notes:
- Prior + train splits are merged before joining with orders.
- `order_number` is converted to `order_idx = order_number - 1`.

Processed stats (baskets.parquet):
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

Processed stats (baskets.parquet):
- Rows: 795,321
- Users: 29,197
- Items: 15,743
- Baskets per user: min 1 / p25 1 / median 2 / mean 3.97 / p75 5 / max 86
- Items per basket: min 1 / p25 2 / median 5 / mean 6.87 / p75 9 / max 111

## Dunnhumby (sample ZIP)

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

Processed stats (baskets.parquet):
- Rows: 25,193,229
- Users: 48,125
- Items: 4,997
- Baskets per user: min 1 / p25 8 / median 41 / mean 78.75 / p75 115 / max 1,157
- Items per basket: min 1 / p25 2 / median 4 / mean 6.65 / p75 9 / max 101

## Dataset-Specific Schemas

### Instacart

**items.parquet**
```
item_id         int32
product_id_raw  int64
product_name    str
aisle_id        int32
aisle           str
department_id   int32
department      str
```

**basket_meta.parquet**
```
user_id                int32
order_idx              int32
order_dow              int32
order_hour_of_day      int32
days_since_prior_order float32
```

**basket_items.parquet**
```
user_id           int32
order_idx         int32
item_id           int32
add_to_cart_order int32
reordered         int32
```

### TaFeng

**items.parquet**
```
item_id          int32
product_id_raw   str
product_subclass int32
```

**user_meta.parquet**
```
user_id   int32
AGE_GROUP str
PIN_CODE  str
```

**basket_meta.parquet**
```
user_id   int32
order_idx int32
date      date
```

**basket_items.parquet**
```
user_id     int32
order_idx   int32
item_id     int32
AMOUNT      float32
ASSET       float32
SALES_PRICE float32
```

### Dunnhumby (sample)

**items.parquet**
```
item_id       int32
prod_code_raw str
prod_code_10  str
prod_code_20  str
prod_code_30  str
prod_code_40  str
```

**basket_meta.parquet**
```
user_id   int32
order_idx int32
SHOP_DATE int64
```

**basket_items.parquet**
```
user_id   int32
order_idx int32
item_id   int32
```

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
