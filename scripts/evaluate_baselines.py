"""Evaluate frequency baselines on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from nbr.data.split import split_user_baskets


def recall_at_k(predicted: list[int], ground_truth: list[int], k: int) -> float:
    """Recall@K for a single user."""
    if not ground_truth:
        return 0.0
    pred_k = set(predicted[:k])
    gt = set(ground_truth)
    return len(pred_k & gt) / len(gt)


def repeat_recall_at_k(
    predicted: list[int],
    ground_truth: list[int],
    history_items: set[int],
    k: int,
) -> float:
    """Repeat Recall@K: only ground truth items seen in history."""
    repeat_gt = [i for i in ground_truth if i in history_items]
    return recall_at_k(predicted, repeat_gt, k)


def explore_recall_at_k(
    predicted: list[int],
    ground_truth: list[int],
    history_items: set[int],
    k: int,
) -> float:
    """Explore Recall@K: only ground truth items NOT seen in history."""
    explore_gt = [i for i in ground_truth if i not in history_items]
    return recall_at_k(predicted, explore_gt, k)


def _build_user_targets(df: pl.DataFrame) -> dict[int, list[int]]:
    """Group items per user (unique) for targets."""
    targets: dict[int, list[int]] = {}
    grouped = df.group_by("user_id").agg(pl.col("item_id").unique().alias("items"))
    for row in grouped.iter_rows(named=True):
        targets[int(row["user_id"])] = [int(i) for i in row["items"]]
    return targets


def _build_personal_counts(df: pl.DataFrame) -> dict[int, dict[int, int]]:
    """Compute per-user item frequency counts from train data."""
    counts: dict[int, dict[int, int]] = {}
    grouped = df.group_by(["user_id", "item_id"]).agg(pl.len().alias("cnt"))
    for row in grouped.iter_rows(named=True):
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        cnt = int(row["cnt"])
        counts.setdefault(uid, {})[iid] = cnt
    return counts


def _evaluate_global(
    global_topk: list[int],
    test_targets: dict[int, list[int]],
    history_items: dict[int, set[int]],
    k_values: list[int],
    user_ids: list[int],
) -> dict[str, float]:
    """Evaluate global baseline (same prediction for all users)."""
    metrics: dict[str, float] = {}
    for k in k_values:
        recall_scores = []
        repeat_scores = []
        explore_scores = []

        preds = global_topk[:k]
        for uid in user_ids:
            target = test_targets.get(uid, [])
            history = history_items.get(uid, set())
            recall_scores.append(recall_at_k(preds, target, k))
            repeat_scores.append(repeat_recall_at_k(preds, target, history, k))
            explore_scores.append(explore_recall_at_k(preds, target, history, k))

        metrics[f"recall@{k}"] = float(sum(recall_scores) / max(len(recall_scores), 1))
        metrics[f"repeat_recall@{k}"] = float(sum(repeat_scores) / max(len(repeat_scores), 1))
        metrics[f"explore_recall@{k}"] = float(sum(explore_scores) / max(len(explore_scores), 1))
    return metrics


def _evaluate_personal(
    personal_counts: dict[int, dict[int, int]],
    test_targets: dict[int, list[int]],
    k_values: list[int],
    user_ids: list[int],
) -> dict[str, float]:
    """Evaluate personal baseline (per-user top-k)."""
    metrics: dict[str, float] = {}
    for k in k_values:
        recall_scores = []
        repeat_scores = []
        explore_scores = []

        for uid in user_ids:
            target = test_targets.get(uid, [])
            counts = personal_counts.get(uid, {})
            history = set(counts.keys())
            preds = [i for i, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]]

            recall_scores.append(recall_at_k(preds, target, k))
            repeat_scores.append(repeat_recall_at_k(preds, target, history, k))
            explore_scores.append(explore_recall_at_k(preds, target, history, k))

        metrics[f"recall@{k}"] = float(sum(recall_scores) / max(len(recall_scores), 1))
        metrics[f"repeat_recall@{k}"] = float(sum(repeat_scores) / max(len(repeat_scores), 1))
        metrics[f"explore_recall@{k}"] = float(sum(explore_scores) / max(len(explore_scores), 1))
    return metrics


def _evaluate_gp(
    global_counts: dict[int, int],
    global_topk: list[int],
    personal_counts: dict[int, dict[int, int]],
    test_targets: dict[int, list[int]],
    k_values: list[int],
    user_ids: list[int],
    alpha: float,
) -> dict[str, float]:
    """Evaluate GP-TopFreq hybrid baseline.

    Candidate set for each user = global_topk union personal items.
    """
    metrics: dict[str, float] = {}
    for k in k_values:
        recall_scores = []
        repeat_scores = []
        explore_scores = []

        global_candidates = global_topk[:k]
        for uid in user_ids:
            target = test_targets.get(uid, [])
            counts = personal_counts.get(uid, {})
            history = set(counts.keys())

            candidates = set(global_candidates) | set(history)
            scored = []
            for item in candidates:
                g = global_counts.get(item, 0)
                p = counts.get(item, 0)
                score = alpha * g + (1.0 - alpha) * p
                scored.append((item, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            preds = [item for item, _ in scored[:k]]

            recall_scores.append(recall_at_k(preds, target, k))
            repeat_scores.append(repeat_recall_at_k(preds, target, history, k))
            explore_scores.append(explore_recall_at_k(preds, target, history, k))

        metrics[f"recall@{k}"] = float(sum(recall_scores) / max(len(recall_scores), 1))
        metrics[f"repeat_recall@{k}"] = float(sum(repeat_scores) / max(len(repeat_scores), 1))
        metrics[f"explore_recall@{k}"] = float(sum(explore_scores) / max(len(explore_scores), 1))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["instacart", "dunnhumby", "tafeng"])
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    processed_dir = Path("data/processed") / args.dataset
    df = pl.read_parquet(processed_dir / "baskets.parquet")

    train_df, val_df, test_df = split_user_baskets(df)

    # Build aggregated statistics for fast evaluation
    personal_counts = _build_personal_counts(train_df)
    test_targets = _build_user_targets(test_df)
    user_ids = list(test_targets.keys())
    if args.max_users is not None:
        user_ids = user_ids[: args.max_users]

    # Precompute global counts and ranking
    global_counts: dict[int, int] = {}
    grouped = train_df.group_by("item_id").agg(pl.len().alias("cnt"))
    for row in grouped.iter_rows(named=True):
        global_counts[int(row["item_id"])] = int(row["cnt"])
    global_topk = [i for i, _ in sorted(global_counts.items(), key=lambda x: x[1], reverse=True)]

    history_items = {uid: set(personal_counts.get(uid, {}).keys()) for uid in user_ids}

    results = {
        "GlobalTopFreq": _evaluate_global(
            global_topk, test_targets, history_items, args.k, user_ids
        ),
        "PersonalTopFreq": _evaluate_personal(personal_counts, test_targets, args.k, user_ids),
        "GPTopFreq": _evaluate_gp(
            global_counts,
            global_topk,
            personal_counts,
            test_targets,
            args.k,
            user_ids,
            args.alpha,
        ),
    }

    # Print results
    print(f"Dataset: {args.dataset} | users: {len(user_ids)}")
    print("=" * 72)
    for name, metrics in results.items():
        print(f"{name}:")
        for key, val in metrics.items():
            print(f"  {key:<20} {val:.4f}")
        print()


if __name__ == "__main__":
    main()
