# CONTEXT

## Current State
We have successfully transitioned the **Vanilla NBR** training from a simple many-to-one setup to a **full causal sequence-to-sequence (GPT-style)** supervision. This provides a much stronger learning signal as the model now learns to predict the next basket at every step in a user's history.

We have also stabilized the training metrics by correcting the loss normalization and fixing the checkpointing logic. Initial evaluation of the vanilla baseline on Instacart (Epoch 4) shows a **Recall@10 of 3.54%** and **NDCG@10 of 5.14%**.

## Just Completed
- [x] Implementation of sequential target generation in `BasketSequenceDataset`.
- [x] Refactoring of `VanillaNBR.loss` to use causal sequence supervision and item-wise averaging.
- [x] Creation of `scripts/evaluate_vanilla.py` for checkpoint verification.
- [x] Successful training run with stabilized loss (standardized range ~0.0016).

## Next Task (Phase 3)
The next step is to move toward **Importance Scoring** (T3.1). We will use the trained Vanilla encoder to compute item importance scores based on how much the basket representation (CLS) shifts when an item is removed.

## Best Metrics (Vanilla @ Instacart)
- **Recall@10**: 0.0354
- **NDCG@10**: 0.0514
- **Repeat Recall@10**: 0.0457
- **Explore Recall@10**: 0.0135
- **Status**: Training is healthy and loss is decreasing.

## Known Issues / Decisions
- **Decision**: End-to-end training is active; gradients flow back to the intra-basket encoder (BERT part).
- **Decision**: Evaluation metrics (Recall/NDCG) are computed specifically on the **last** prediction in the sequence to maintain standard NBR benchmark compatibility.
