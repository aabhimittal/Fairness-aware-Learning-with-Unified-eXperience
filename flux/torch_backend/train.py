"""Training loop and synthetic interaction data for the torch backend."""

from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    dim: int = 16
    curvature: float = 1.0
    lr: float = 0.05
    epochs: int = 30
    batch_size: int = 256
    margin: float = 1.0
    seed: int = 0


def make_synthetic_interactions(n_users: int = 50, n_items: int = 200,
                                n_topics: int = 5, per_user: int = 30,
                                seed: int = 0):
    """Hierarchical synthetic data: items belong to topics, users prefer
    1-2 topics. Returns (users, pos_items, item_topics) tensors.

    The topic structure is tree-like on purpose — it is exactly the regime
    where hyperbolic embeddings should beat Euclidean ones.
    """
    g = torch.Generator().manual_seed(seed)
    item_topics = torch.randint(0, n_topics, (n_items,), generator=g)
    users, positives = [], []
    for u in range(n_users):
        fav = torch.randint(0, n_topics, (2,), generator=g)
        pool = torch.nonzero(
            (item_topics == fav[0]) | (item_topics == fav[1])
        ).flatten()
        picks = pool[torch.randint(0, len(pool), (per_user,), generator=g)]
        users.append(torch.full((per_user,), u))
        positives.append(picks)
    return torch.cat(users), torch.cat(positives), item_topics


def train(model, users: torch.Tensor, positives: torch.Tensor,
          n_items: int, config: TrainConfig, verbose: bool = False):
    """Train any model exposing triplet_loss(users, pos, neg, margin).

    Negatives are sampled uniformly per batch. Returns per-epoch losses.
    """
    torch.manual_seed(config.seed)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = len(users)
    losses = []
    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            neg = torch.randint(0, n_items, (len(idx),))
            loss = model.triplet_loss(users[idx], positives[idx], neg,
                                      margin=config.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        losses.append(total / n)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: loss {losses[-1]:.4f}")
    return losses
