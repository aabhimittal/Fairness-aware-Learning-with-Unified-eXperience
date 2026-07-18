# 🌊 FLUX: Fairness-aware Learning with Unified eXperience

[![CI](https://github.com/aabhimittal/Fairness-aware-Learning-with-Unified-eXperience/actions/workflows/ci.yml/badge.svg)](https://github.com/aabhimittal/Fairness-aware-Learning-with-Unified-eXperience/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

A novel AI-powered social media ranking system that combines **hyperbolic
geometry**, **quantum-inspired attention**, **causal inference**, and
**fairness-aware re-ranking** into one pipeline.

## The core idea

Think of user interests like galaxies in expanding space. Traditional
algorithms embed users and content in flat Euclidean space, but interests are
*hierarchical* — loving "dogs" is broader than loving "golden retrievers".
Hyperbolic space (the Poincaré ball) grows exponentially toward its boundary,
naturally fitting tree-like taxonomies: broad interests sit near the origin,
niche interests near the edge.

On top of that, a user's interests are held as a **quantum-style
superposition** — complex amplitudes that *collapse* into a concrete interest
profile only when measured against a context (time of day, session, social
signals). Complex phases let interests interfere constructively or
destructively, something a plain softmax can't express.

## Installation

```bash
pip install flux-ranking                 # NumPy core
pip install "flux-ranking[torch]"        # + PyTorch training backend
pip install "flux-ranking[api]"          # + FastAPI serving layer
pip install "flux-ranking[eval]"         # + everything needed to run the notebooks
```

📚 **Full documentation**: concepts, quickstart, evaluation, serving, and
release guides live in [`docs/`](docs/index.md) (MkDocs site, auto-deployed
to GitHub Pages from `main`).

From source:

```bash
git clone https://github.com/aabhimittal/Fairness-aware-Learning-with-Unified-eXperience
cd Fairness-aware-Learning-with-Unified-eXperience
pip install -e ".[eval]"
```

## Pipeline

```
content pool
   │
   ▼
Hyperbolic relevance ──── geodesic proximity on the Poincaré ball   (flux/hyperbolic.py)
   │
   ▼
Quantum attention ─────── context collapses interest superposition  (flux/attention.py)
   │
   ▼
Causal debiasing ──────── IPS removes position/exposure bias        (flux/causal.py)
   │
   ▼
Fair re-ranking ───────── amortized exposure parity across creators (flux/fairness.py)
   │
   ▼
served slate                                                        (flux/engine.py)
```

| Module | Novel component | Why it matters |
|---|---|---|
| `hyperbolic.py` | Poincaré-ball embeddings + Riemannian SGD | Represents interest hierarchies with far fewer dimensions than Euclidean space |
| `attention.py` | Complex-amplitude interest state with Born-rule collapse | Same user, different context → different feed; interference models conflicting interests |
| `causal.py` | Clipped inverse propensity scoring | Separates "clicked because good" from "clicked because shown first" |
| `fairness.py` | Greedy amortized exposure-parity re-ranker | Breaks the rich-get-richer loop; emerging creators get their target share of exposure |
| `engine.py` | Unified `FluxEngine` | One `rank()` call runs the whole pipeline |
| `datasets.py` | MovieLens 100K loader | Cached download, leave-latest-out split, popularity groups for fairness studies |
| `torch_backend/` | Differentiable Poincaré manifold + trainable models | Autograd training with plain Adam via tangent-space parametrization |

## Quick start

```python
import numpy as np
from flux import FluxEngine, Content

engine = FluxEngine(
    n_interests=3,
    target_shares={"established": 0.55, "emerging": 0.45},
    fairness_weight=2.0,
)
slate = engine.rank(user_point, content_pool, context=np.array([1.0, 2.0, 0.3]))
engine.feedback(np.array([0.2, 1.0, -0.3]))  # engagement updates the quantum state
```

Training on real data:

```python
import torch
from flux.datasets import load_ml100k
from flux.torch_backend import HyperbolicInterestModel, TrainConfig, train

ml = load_ml100k()                       # downloads once (~5 MB), then cached
model = HyperbolicInterestModel(ml.n_users, ml.n_items, dim=16)
train(model,
      torch.from_numpy(ml.train[:, 0]), torch.from_numpy(ml.train[:, 1]),
      n_items=ml.n_items, config=TrainConfig(epochs=40))
top10 = model.rank_items(user=0, k=10)
```

## Evaluations

Two executed notebooks (outputs committed):

- **[notebooks/evaluation.ipynb](notebooks/evaluation.ipynb)** — synthetic
  hierarchical data: NDCG@10/Hit@10 vs a Euclidean baseline, exposure
  fairness before/after `FairReranker`, and IPS vs naive CTR under simulated
  position bias (IPS recovers 0.387 of a true 0.4 CTR where naive collapses
  to 0.155).
- **[notebooks/movielens_evaluation.ipynb](notebooks/movielens_evaluation.ipynb)**
  — **real data** (MovieLens 100K, leave-latest-out): the hyperbolic model
  beats the equal-dimension Euclidean baseline (HR@10 0.077 vs 0.067, NDCG@10
  0.038 vs 0.031), and fair re-ranking lifts long-tail movie exposure from
  5% to 42% against a 50% target using the same trained scores.

Run the demo and tests:

```bash
python examples/demo.py
python -m unittest discover tests -v      # 30 tests; torch tests skip if torch absent
```

## Project structure

```
flux/                  core NumPy package
  torch_backend/       optional PyTorch training backend
notebooks/             executed evaluation notebooks
examples/demo.py       end-to-end serving demo
tests/                 unit tests (offline; dataset parser tested on fixtures)
pyproject.toml         packaging (pip install flux-ranking)
.github/workflows/     CI: tests on 3.10/3.12 + package build check
```

## Design notes

- **Optional heavy deps** — the core is pure NumPy; `torch` ships as an
  extra so the math stays transparent and the install light.
- **Amortized fairness** — the re-ranker tracks cumulative exposure *across
  requests*, so parity holds over time, not just per slate.
- **Bounded-variance debiasing** — IPS weights are clipped, trading a little
  bias for stable estimates on sparse logs.
- **Trainable geometry** — the torch backend parametrizes points in tangent
  space at the origin, so any Euclidean optimizer trains hyperbolic
  embeddings without Riemannian machinery.

## License

MIT — see [LICENSE](LICENSE). MovieLens data is subject to the
[GroupLens terms of use](https://grouplens.org/datasets/movielens/).
