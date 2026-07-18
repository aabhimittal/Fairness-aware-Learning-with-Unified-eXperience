"""Causal debiasing of engagement signals.

Observed clicks are confounded by exposure: content shown at rank 1 gets
clicked more regardless of quality. Inverse propensity scoring (IPS)
reweights each observed interaction by 1/P(shown), recovering an unbiased
estimate of true preference from logged feedback.
"""

import numpy as np


class IPSDebiaser:
    def __init__(self, position_decay: float = 0.7, clip: float = 20.0):
        """position_decay: P(examined at rank k) ~ decay^k; clip caps weights
        to bound variance (clipped IPS)."""
        if not 0 < position_decay <= 1:
            raise ValueError("position_decay must be in (0, 1]")
        self.position_decay = position_decay
        self.clip = clip

    def propensity(self, rank: np.ndarray) -> np.ndarray:
        """Examination probability for 0-indexed ranks."""
        return self.position_decay ** np.asarray(rank, dtype=float)

    def debias(self, clicks: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        """Unbiased per-item preference estimates from logged (click, rank) pairs.

        Returns click / propensity, clipped: an estimate of how often the item
        would be clicked if every item were examined equally.
        """
        clicks = np.asarray(clicks, dtype=float)
        weights = np.minimum(1.0 / self.propensity(ranks), self.clip)
        return clicks * weights

    def estimate_ctr(self, clicks: np.ndarray, ranks: np.ndarray) -> float:
        """Self-normalized IPS estimate of the true click-through rate."""
        clicks = np.asarray(clicks, dtype=float)
        w = np.minimum(1.0 / self.propensity(ranks), self.clip)
        return float(np.sum(clicks * w) / np.sum(w))
