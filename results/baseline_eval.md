# Baseline Evaluation Results

This file captures how the frequency baselines were run and the resulting metrics.
All runs used the same preprocessing pipeline and the same train/val/test split rule
(last basket=test, second-to-last=val, remainder=train).

NOTE: These are **subset runs** (2,000 users). Full-dataset evaluation is still required
for final reporting. GPTopFreq alpha sweeps have NOT been run yet.

## Run Configuration

- Script: `scripts/evaluate_baselines.py`
- Metrics: Recall@K, Repeat Recall@K, Explore Recall@K for K ∈ {5,10,20}
- Baselines: GlobalTopFreq, PersonalTopFreq, GPTopFreq (alpha=0.5)
- Subset: `--max-users 2000` (to keep runtime reasonable)
- Full-dataset runs: **pending**
- GPTopFreq alpha sweeps: **pending**

## Commands

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset instacart --k 5 10 20 --max-users 2000
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset tafeng --k 5 10 20 --max-users 2000
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset dunnhumby --k 5 10 20 --max-users 2000
```

## Results

### Instacart (2,000 users)

GlobalTopFreq:
- recall@5 0.0499
- repeat_recall@5 0.0654
- explore_recall@5 0.0215
- recall@10 0.0712
- repeat_recall@10 0.0951
- explore_recall@10 0.0326
- recall@20 0.0981
- repeat_recall@20 0.1261
- explore_recall@20 0.0511

PersonalTopFreq:
- recall@5 0.2208
- repeat_recall@5 0.3583
- explore_recall@5 0.0000
- recall@10 0.3063
- repeat_recall@10 0.5065
- explore_recall@10 0.0000
- recall@20 0.3975
- repeat_recall@20 0.6674
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0499
- repeat_recall@5 0.0654
- explore_recall@5 0.0215
- recall@10 0.0712
- repeat_recall@10 0.0951
- explore_recall@10 0.0326
- recall@20 0.0981
- repeat_recall@20 0.1261
- explore_recall@20 0.0511

### TaFeng (2,000 users)

GlobalTopFreq:
- recall@5 0.0496
- repeat_recall@5 0.0299
- explore_recall@5 0.0360
- recall@10 0.0582
- repeat_recall@10 0.0344
- explore_recall@10 0.0439
- recall@20 0.0820
- repeat_recall@20 0.0446
- explore_recall@20 0.0628

PersonalTopFreq:
- recall@5 0.0353
- repeat_recall@5 0.0916
- explore_recall@5 0.0000
- recall@10 0.0479
- repeat_recall@10 0.1276
- explore_recall@10 0.0000
- recall@20 0.0594
- repeat_recall@20 0.1654
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0496
- repeat_recall@5 0.0299
- explore_recall@5 0.0360
- recall@10 0.0582
- repeat_recall@10 0.0344
- explore_recall@10 0.0439
- recall@20 0.0820
- repeat_recall@20 0.0446
- explore_recall@20 0.0628

### Dunnhumby (2,000 users)

GlobalTopFreq:
- recall@5 0.0834
- repeat_recall@5 0.0778
- explore_recall@5 0.0109
- recall@10 0.1059
- repeat_recall@10 0.0968
- explore_recall@10 0.0153
- recall@20 0.1326
- repeat_recall@20 0.1218
- explore_recall@20 0.0183

PersonalTopFreq:
- recall@5 0.2011
- repeat_recall@5 0.2130
- explore_recall@5 0.0000
- recall@10 0.2838
- repeat_recall@10 0.2988
- explore_recall@10 0.0000
- recall@20 0.3780
- repeat_recall@20 0.3980
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0834
- repeat_recall@5 0.0778
- explore_recall@5 0.0109
- recall@10 0.1059
- repeat_recall@10 0.0968
- explore_recall@10 0.0153
- recall@20 0.1326
- repeat_recall@20 0.1218
- explore_recall@20 0.0183
