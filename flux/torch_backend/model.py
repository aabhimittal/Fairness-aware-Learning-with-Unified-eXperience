"""Hyperbolic user/content embedding model."""

import torch
from torch import nn

from .manifold import PoincareManifold


class HyperbolicInterestModel(nn.Module):
    """Embeds users and content items on a shared Poincare ball.

    Score of (user, item) = -geodesic distance, so closer = more relevant.
    Parameters are stored in tangent space at the origin and mapped onto the
    ball with expmap0, which keeps optimization unconstrained (any Euclidean
    optimizer works) while distances remain hyperbolic.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 16,
                 c: float = 1.0, init_scale: float = 1e-2):
        super().__init__()
        self.manifold = PoincareManifold(c)
        self.user_tangent = nn.Embedding(n_users, dim)
        self.item_tangent = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_tangent.weight, std=init_scale)
        nn.init.normal_(self.item_tangent.weight, std=init_scale)

    def user_points(self, users: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap0(self.user_tangent(users))

    def item_points(self, items: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap0(self.item_tangent(items))

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return -self.manifold.dist(self.user_points(users), self.item_points(items))

    def triplet_loss(self, users: torch.Tensor, pos: torch.Tensor,
                     neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """BPR-style margin loss: pos item should score above neg by `margin`."""
        d_pos = -self.score(users, pos)
        d_neg = -self.score(users, neg)
        return torch.relu(margin + d_pos - d_neg).mean()

    @torch.no_grad()
    def rank_items(self, user: int, k: int | None = None) -> torch.Tensor:
        """Indices of all items for one user, best first."""
        u = self.user_points(torch.tensor([user])).expand(
            self.item_tangent.num_embeddings, -1
        )
        all_items = self.manifold.expmap0(self.item_tangent.weight)
        order = self.manifold.dist(u, all_items).argsort()
        return order if k is None else order[:k]


class EuclideanBaseline(nn.Module):
    """Same architecture with dot-product scores, for apples-to-apples evals."""

    def __init__(self, n_users: int, n_items: int, dim: int = 16,
                 init_scale: float = 1e-2):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=init_scale)
        nn.init.normal_(self.item_emb.weight, std=init_scale)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)

    def triplet_loss(self, users: torch.Tensor, pos: torch.Tensor,
                     neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        return torch.relu(
            margin - self.score(users, pos) + self.score(users, neg)
        ).mean()

    @torch.no_grad()
    def rank_items(self, user: int, k: int | None = None) -> torch.Tensor:
        scores = self.item_emb.weight @ self.user_emb.weight[user]
        order = scores.argsort(descending=True)
        return order if k is None else order[:k]
