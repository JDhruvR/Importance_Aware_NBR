"""Frequency baselines for Next Basket Recommendation.

All predictors implement:
    predict(user_history: list[list[int]]) -> list[int]
where user_history is a sequence of baskets (each basket is list of item IDs).
"""

from __future__ import annotations

from collections import Counter


class GlobalTopFreq:
    """Recommend globally most frequent items."""

    def __init__(self, topk: int = 10) -> None:
        self.topk = topk
        self._global_rank: list[int] = []

    def fit(self, all_histories: list[list[list[int]]]) -> None:
        """Compute global item frequencies from all users.

        Args:
            all_histories: list of user histories, each a list of baskets.
        """
        counter: Counter[int] = Counter()
        for history in all_histories:
            for basket in history:
                counter.update(basket)
        self._global_rank = [item for item, _ in counter.most_common()]

    def predict(self, user_history: list[list[int]]) -> list[int]:
        """Return top-k global frequent items (ignores history content)."""
        return self._global_rank[: self.topk]


class PersonalTopFreq:
    """Recommend user's most frequent items in history."""

    def __init__(self, topk: int = 10) -> None:
        self.topk = topk

    def predict(self, user_history: list[list[int]]) -> list[int]:
        """Return top-k items by personal frequency."""
        counter: Counter[int] = Counter()
        for basket in user_history:
            counter.update(basket)
        return [item for item, _ in counter.most_common(self.topk)]


class GPTopFreq:
    """Hybrid of Global and Personal TopFreq from Li et al. 2023.

    Score(item) = alpha * GlobalFreq(item) + (1 - alpha) * PersonalFreq(item)
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.topk = topk
        self.alpha = alpha
        self._global_counts: Counter[int] = Counter()

    def fit(self, all_histories: list[list[list[int]]]) -> None:
        """Compute global item frequencies from all users.

        Args:
            all_histories: list of user histories, each a list of baskets.
        """
        counter: Counter[int] = Counter()
        for history in all_histories:
            for basket in history:
                counter.update(basket)
        self._global_counts = counter

    def predict(self, user_history: list[list[int]]) -> list[int]:
        """Return top-k items by the GP-TopFreq hybrid score."""
        personal = Counter()
        for basket in user_history:
            personal.update(basket)

        # Union of items that appear globally or personally
        items = set(self._global_counts.keys()) | set(personal.keys())
        scored = []
        for item in items:
            g = self._global_counts.get(item, 0)
            p = personal.get(item, 0)
            score = self.alpha * g + (1.0 - self.alpha) * p
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[: self.topk]]
