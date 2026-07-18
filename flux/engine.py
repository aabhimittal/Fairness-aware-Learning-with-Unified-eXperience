"""FluxEngine: the unified ranking pipeline.

score = quantum-context-weighted hyperbolic relevance
      -> causally debiased by historical exposure
      -> fairness-aware re-ranked slate
"""

from dataclasses import dataclass, field

import numpy as np

from .attention import QuantumAttention
from .causal import IPSDebiaser
from .fairness import FairReranker
from .hyperbolic import PoincareBall


@dataclass
class Content:
    id: str
    creator_group: str  # e.g. "emerging" | "established"
    embedding: np.ndarray  # point on the Poincare ball
    interest_affinity: np.ndarray  # affinity to each interest dimension
    logged_clicks: float = 0.0
    logged_rank: int = 0
    metadata: dict = field(default_factory=dict)


class FluxEngine:
    def __init__(self, n_interests: int, target_shares: dict,
                 curvature: float = 1.0, fairness_weight: float = 1.0,
                 seed: int = 0):
        self.ball = PoincareBall(curvature)
        self.attention = QuantumAttention(n_interests, seed=seed)
        self.debiaser = IPSDebiaser()
        self.reranker = FairReranker(target_shares, fairness_weight)

    def relevance(self, user_point: np.ndarray, item: Content,
                  interest_weights: np.ndarray) -> float:
        """Blend geometric proximity with context-collapsed interest match."""
        geo = float(np.exp(-self.ball.dist(user_point, item.embedding)))
        interest = float(interest_weights @ item.interest_affinity)
        return geo * (0.5 + 0.5 * interest)

    def rank(self, user_point: np.ndarray, items: list, context: np.ndarray,
             k: int | None = None) -> list:
        """Produce a fair slate of Content for one user and context.

        Returns items in serving order.
        """
        weights = self.attention.collapse(context)
        raw = np.array([self.relevance(user_point, it, weights) for it in items])
        # correct raw scores by debiased engagement: items whose true (IPS)
        # appeal exceeds their logged appeal get boosted, and vice versa
        clicks = np.array([it.logged_clicks for it in items])
        ranks = np.array([it.logged_rank for it in items])
        debiased = self.debiaser.debias(clicks, ranks)
        appeal = debiased / (debiased.max() + 1e-9) if debiased.max() > 0 else debiased
        scores = raw * (1.0 + appeal)
        order = self.reranker.rerank(scores, [it.creator_group for it in items], k=k)
        return [items[i] for i in order]

    def feedback(self, engagement: np.ndarray):
        """Fold a per-interest engagement signal back into the user state."""
        self.attention.evolve(engagement)
