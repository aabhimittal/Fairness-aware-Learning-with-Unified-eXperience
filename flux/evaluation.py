"""Reusable evaluation pipeline (requires the torch extra).

Everything the evaluation notebooks demonstrate is callable from here (and
from the `flux-eval` CLI) — the notebooks are demos, not the implementation.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from .datasets import LOADERS, MovieLens
from .fairness import FairReranker


@dataclass
class EvalConfig:
    dataset: str = "ml-100k"  # key into flux.datasets.LOADERS
    dim: int = 16
    epochs: int = 40
    lr: float = 0.05
    batch_size: int = 1024
    seed: int = 0
    k: int = 10
    head_fraction: float = 0.2
    target_shares: dict = field(
        default_factory=lambda: {"head": 0.5, "tail": 0.5})
    fairness_weight: float = 5.0
    fairness_users: int = 300  # users to serve in the exposure study


def ranking_metrics(model, data: MovieLens, k: int = 10) -> dict:
    """Leave-latest-out HR@k and NDCG@k for any model with rank_items()."""
    train_sets: dict[int, set] = {}
    for u, i in data.train:
        train_sets.setdefault(int(u), set()).add(int(i))
    hits, ndcgs = [], []
    for u, target in data.test:
        order = model.rank_items(int(u)).numpy()
        seen = train_sets.get(int(u), set())
        order = order[[i not in seen for i in order]]
        pos = int(np.nonzero(order == target)[0][0])
        hits.append(1.0 if pos < k else 0.0)
        ndcgs.append(1.0 / np.log2(pos + 2) if pos < k else 0.0)
    return {f"hr@{k}": float(np.mean(hits)),
            f"ndcg@{k}": float(np.mean(ndcgs))}


def exposure_report(model, data: MovieLens, config: EvalConfig) -> dict:
    """Head/tail exposure shares before vs. after fair re-ranking."""
    import torch

    groups = list(data.popularity_group(config.head_fraction))
    item_idx = torch.arange(data.n_items)
    n_users = min(config.fairness_users, data.n_users)

    def serve(weight: float) -> dict:
        rr = FairReranker(config.target_shares, fairness_weight=weight)
        for u in range(n_users):
            scores = -model.manifold.dist(
                model.user_points(torch.tensor([u])).expand(data.n_items, -1),
                model.item_points(item_idx)).detach().numpy()
            rr.rerank(scores, groups, k=config.k)
        return {g: round(s, 4) for g, s in rr.exposure_shares().items()}

    return {"before": serve(0.0), "after": serve(config.fairness_weight),
            "target": config.target_shares}


def run_evaluation(config: EvalConfig | None = None,
                   verbose: bool = True) -> dict:
    """Full pipeline: load dataset, train hyperbolic + Euclidean models,
    compute ranking metrics and the fairness exposure report."""
    import torch

    from .torch_backend import HyperbolicInterestModel, TrainConfig, train
    from .torch_backend.model import EuclideanBaseline

    config = config or EvalConfig()
    if config.dataset not in LOADERS:
        raise ValueError(
            f"unknown dataset {config.dataset!r}; options: {sorted(LOADERS)}")
    data = LOADERS[config.dataset]()
    if verbose:
        print(f"{config.dataset}: users={data.n_users} items={data.n_items} "
              f"train={len(data.train)} test={len(data.test)}")

    users = torch.from_numpy(data.train[:, 0])
    positives = torch.from_numpy(data.train[:, 1])
    tc = TrainConfig(dim=config.dim, epochs=config.epochs, lr=config.lr,
                     batch_size=config.batch_size, seed=config.seed)

    torch.manual_seed(config.seed)
    hyp = HyperbolicInterestModel(data.n_users, data.n_items, dim=config.dim)
    euc = EuclideanBaseline(data.n_users, data.n_items, dim=config.dim)
    hyp_losses = train(hyp, users, positives, data.n_items, tc, verbose=verbose)
    euc_losses = train(euc, users, positives, data.n_items, tc)

    results = {
        "config": asdict(config),
        "hyperbolic": ranking_metrics(hyp, data, config.k),
        "euclidean": ranking_metrics(euc, data, config.k),
        "final_loss": {"hyperbolic": hyp_losses[-1], "euclidean": euc_losses[-1]},
        "fairness": exposure_report(hyp, data, config),
    }
    if verbose:
        print(f"hyperbolic: {results['hyperbolic']}")
        print(f"euclidean : {results['euclidean']}")
        print(f"fairness  : {results['fairness']}")
    return results
