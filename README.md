# Importance-Aware Next Basket Recommendation

This repo implements baselines and model variants from:
"Not All Items Are Created Equal: Importance-Aware Next Basket Recommendation".

## Data Setup (Processed Only)

The `data/processed/` folder is **not tracked** in git. I will provide a zip of the
processed data. To use it:

1. Download the processed zip provided by the project owner.
2. Unzip it into the repo so the folder structure becomes:

```
data/processed/{instacart,tafeng,dunnhumby}/
```

Raw data is **not required** for training or evaluation once processed data is available.

## Quick Start

Run frequency baselines (example):

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset instacart --k 5 10 20 --max-users 2000
```
