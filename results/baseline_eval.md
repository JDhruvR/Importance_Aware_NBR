# Baseline Evaluation Results

This file captures how the frequency baselines were run and the resulting metrics.
All runs used the same preprocessing pipeline and the same train/val/test split rule
(last basket=test, second-to-last=val, remainder=train).

NOTE: Full-dataset evaluation and GPTopFreq alpha sweeps are complete. The
2,000-user subset runs are retained below for reference.

## Run Configuration

- Script: `scripts/evaluate_baselines.py`
- Metrics: Recall@K, Repeat Recall@K, Explore Recall@K for K ∈ {5,10,20}
- Baselines: GlobalTopFreq, PersonalTopFreq, GPTopFreq
- Full-dataset runs: no `--max-users`
- Subset runs: `--max-users 2000` (reference only)
- GPTopFreq alpha sweep: alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}

## Commands

Full datasets:

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset instacart --k 5 10 20
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset tafeng --k 5 10 20
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset dunnhumby --k 5 10 20
```

GPTopFreq alpha sweeps:

```bash
for alpha in 0 0.25 0.5 0.75 1.0; do PYTHONPATH=. python scripts/evaluate_baselines.py --dataset instacart --k 5 10 20 --alpha $alpha; done
for alpha in 0 0.25 0.5 0.75 1.0; do PYTHONPATH=. python scripts/evaluate_baselines.py --dataset tafeng --k 5 10 20 --alpha $alpha; done
for alpha in 0 0.25 0.5 0.75 1.0; do PYTHONPATH=. python scripts/evaluate_baselines.py --dataset dunnhumby --k 5 10 20 --alpha $alpha; done
```

Subset runs (2,000 users):

```bash
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset instacart --k 5 10 20 --max-users 2000
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset tafeng --k 5 10 20 --max-users 2000
PYTHONPATH=. python scripts/evaluate_baselines.py --dataset dunnhumby --k 5 10 20 --max-users 2000
```

## Full-Dataset Results

### Instacart (206,209 users)

GlobalTopFreq:
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

PersonalTopFreq:
- recall@5 0.2073
- repeat_recall@5 0.3451
- explore_recall@5 0.0000
- recall@10 0.2951
- repeat_recall@10 0.4959
- explore_recall@10 0.0000
- recall@20 0.3884
- repeat_recall@20 0.6583
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

### TaFeng (29,197 users)

GlobalTopFreq:
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0573
- repeat_recall@10 0.0283
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

PersonalTopFreq:
- recall@5 0.0314
- repeat_recall@5 0.0833
- explore_recall@5 0.0000
- recall@10 0.0433
- repeat_recall@10 0.1207
- explore_recall@10 0.0000
- recall@20 0.0554
- repeat_recall@20 0.1597
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0573
- repeat_recall@10 0.0283
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

### Dunnhumby (48,125 users)

GlobalTopFreq:
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1196
- explore_recall@20 0.0235

PersonalTopFreq:
- recall@5 0.1952
- repeat_recall@5 0.2080
- explore_recall@5 0.0000
- recall@10 0.2770
- repeat_recall@10 0.2953
- explore_recall@10 0.0000
- recall@20 0.3752
- repeat_recall@20 0.3992
- explore_recall@20 0.0000

GPTopFreq (alpha=0.5):
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1196
- explore_recall@20 0.0235

## GPTopFreq Alpha Sweeps (Full Datasets)

### Instacart

alpha=0:
- recall@5 0.2078
- repeat_recall@5 0.3463
- explore_recall@5 0.0003
- recall@10 0.2968
- repeat_recall@10 0.4964
- explore_recall@10 0.0024
- recall@20 0.3943
- repeat_recall@20 0.6581
- explore_recall@20 0.0085

alpha=0.25:
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

alpha=0.5:
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

alpha=0.75:
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

alpha=1.0:
- recall@5 0.0470
- repeat_recall@5 0.0641
- explore_recall@5 0.0187
- recall@10 0.0700
- repeat_recall@10 0.0934
- explore_recall@10 0.0312
- recall@20 0.0955
- repeat_recall@20 0.1216
- explore_recall@20 0.0485

### TaFeng

alpha=0:
- recall@5 0.0526
- repeat_recall@5 0.0866
- explore_recall@5 0.0194
- recall@10 0.0739
- repeat_recall@10 0.1251
- explore_recall@10 0.0289
- recall@20 0.1022
- repeat_recall@20 0.1618
- explore_recall@20 0.0469

alpha=0.25:
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0574
- repeat_recall@10 0.0284
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

alpha=0.5:
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0573
- repeat_recall@10 0.0283
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

alpha=0.75:
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0573
- repeat_recall@10 0.0283
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

alpha=1.0:
- recall@5 0.0477
- repeat_recall@5 0.0239
- explore_recall@5 0.0373
- recall@10 0.0573
- repeat_recall@10 0.0283
- explore_recall@10 0.0459
- recall@20 0.0805
- repeat_recall@20 0.0379
- explore_recall@20 0.0662

### Dunnhumby

alpha=0:
- recall@5 0.2048
- repeat_recall@5 0.2086
- explore_recall@5 0.0089
- recall@10 0.2903
- repeat_recall@10 0.2953
- explore_recall@10 0.0132
- recall@20 0.3939
- repeat_recall@20 0.3995
- explore_recall@20 0.0185

alpha=0.25:
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1197
- explore_recall@20 0.0235

alpha=0.5:
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1196
- explore_recall@20 0.0235

alpha=0.75:
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1196
- explore_recall@20 0.0235

alpha=1.0:
- recall@5 0.0852
- repeat_recall@5 0.0789
- explore_recall@5 0.0126
- recall@10 0.1074
- repeat_recall@10 0.0977
- explore_recall@10 0.0176
- recall@20 0.1337
- repeat_recall@20 0.1196
- explore_recall@20 0.0235

## Subset Results (2,000 users)

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
