"""Download and verify raw dataset files for Importance-Aware NBR."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATASETS: dict[str, dict[str, object]] = {
    "instacart": {
        "raw_dir": Path("data/raw/instacart"),
        "expected_files": [
            "orders.csv",
            "order_products__prior.csv",
            "order_products__train.csv",
        ],
        "instructions": (
            "Instacart 2017 — Market Basket Analysis\n"
            "  1. Install kaggle:  pip install kaggle\n"
            "  2. Set up credentials: https://github.com/Kaggle/kaggle-api#api-credentials\n"
            "  3. Run:  kaggle datasets download -d psparks/instacart-market-basket-analysis\n"
            "  4. Unzip into: data/raw/instacart/\n"
        ),
    },
    "dunnhumby": {
        "raw_dir": Path("data/raw/dunnhumby"),
        "expected_files": [
            "transaction_data.csv.gz",
            "product.csv",
            "customer.csv",
        ],
        "instructions": (
            "Dunnhumby — The Complete Journey\n"
            "  1. Visit: https://www.dunnhumby.com/source-files/\n"
            "  2. Download 'The Complete Journey' dataset\n"
            "  3. Place files in: data/raw/dunnhumby/\n"
            "  4. Expected: transaction_data.csv.gz, product.csv, customer.csv\n"
        ),
    },
    "tafeng": {
        "raw_dir": Path("data/raw/tafeng"),
        "expected_files": [
            "ta_feng_all_months.csv",
        ],
        "instructions": (
            "TaFeng — Grocery Dataset\n"
            "  1. Visit: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset\n"
            "  2. Download via kaggle CLI or web interface\n"
            "  3. Unzip into: data/raw/tafeng/\n"
            "  4. Expected: ta_feng_all_months.csv\n"
        ),
    },
}


def _check_dataset(name: str) -> bool:
    """Return True if all expected files exist for *name*."""
    ds = DATASETS[name]
    raw_dir = ds["raw_dir"]
    missing = [f for f in ds["expected_files"] if not (raw_dir / f).exists()]
    if missing:
        print(f"[{name}] MISSING files in {raw_dir}:")
        for f in missing:
            print(f"  - {f}")
        print()
        print(ds["instructions"])
        return False
    print(f"[{name}] OK — all {len(ds['expected_files'])} files found in {raw_dir}")
    return True


def main() -> None:
    """Check all datasets and print instructions for any that are missing."""
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    all_ok = True
    for name in DATASETS:
        if not _check_dataset(name):
            all_ok = False

    if not all_ok:
        print("\nSome datasets are missing. Follow the instructions above to download them.")
        sys.exit(1)
    else:
        print("\nAll datasets present and ready for preprocessing.")


if __name__ == "__main__":
    main()
