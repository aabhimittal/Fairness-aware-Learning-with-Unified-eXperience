"""Fairness-aware exposure re-ranking.

Pure relevance ranking concentrates exposure on already-popular creators
(rich-get-richer). FairReranker greedily rebuilds the slate so that each
creator group's cumulative exposure tracks its target share (amortized
exposure parity), trading at most a bounded amount of relevance per slot.
"""

from collections import defaultdict

import numpy as np

# position-discounted exposure, as in DCG
def _exposure(rank: int) -> float:
    return 1.0 / np.log2(rank + 2.0)


class FairReranker:
    def __init__(self, target_shares: dict, fairness_weight: float = 1.0):
        """target_shares: {group: desired fraction of total exposure}, summing to ~1.
        fairness_weight: 0 = pure relevance, larger = stricter parity."""
        total = sum(target_shares.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"target shares must sum to 1, got {total}")
        self.target_shares = dict(target_shares)
        self.fairness_weight = fairness_weight
        self.cum_exposure = defaultdict(float)  # amortized across calls

    def _deficit(self, group: str) -> float:
        """How under-exposed a group is vs. its target (positive = owed)."""
        total = sum(self.cum_exposure.values())
        if total == 0:
            return self.target_shares.get(group, 0.0)
        share = self.cum_exposure[group] / total
        return self.target_shares.get(group, 0.0) - share

    def rerank(self, scores: np.ndarray, groups: list, k: int | None = None) -> list:
        """Return item indices ordered fairly.

        Greedy: at each slot pick argmax(relevance + weight * group deficit),
        updating cumulative exposure as slots fill.
        """
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(groups):
            raise ValueError("scores and groups must align")
        k = len(scores) if k is None else min(k, len(scores))
        remaining = set(range(len(scores)))
        slate = []
        for slot in range(k):
            best, best_val = None, -np.inf
            for i in remaining:
                val = scores[i] + self.fairness_weight * self._deficit(groups[i])
                if val > best_val:
                    best, best_val = i, val
            slate.append(best)
            remaining.discard(best)
            self.cum_exposure[groups[best]] += _exposure(slot)
        return slate

    def exposure_shares(self) -> dict:
        """Realized cumulative exposure share per group."""
        total = sum(self.cum_exposure.values())
        if total == 0:
            return {}
        return {g: e / total for g, e in self.cum_exposure.items()}
