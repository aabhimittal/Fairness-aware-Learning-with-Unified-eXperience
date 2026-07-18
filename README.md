# 🌊 FLUX: Fairness-aware Learning with Unified eXperience

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

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py          # end-to-end demo, two contexts + fairness report
python -m unittest discover tests -v
```

### PyTorch training backend (optional)

```bash
pip install -r requirements-torch.txt
```

`flux/torch_backend/` provides a differentiable Poincaré manifold
(autograd replaces the core's finite-difference gradients), a
`HyperbolicInterestModel` that embeds users and content on the ball via
tangent-space parametrization (so plain Adam works), a `EuclideanBaseline`
for comparisons, and a `train()` loop with `TrainConfig` +
`make_synthetic_interactions()` for hierarchical synthetic data.

See **[notebooks/evaluation.ipynb](notebooks/evaluation.ipynb)** for a full
executed evaluation: NDCG@10 / Hit@10 vs the Euclidean baseline, exposure
fairness before/after `FairReranker`, and IPS vs naive CTR under simulated
position bias.

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

## Design notes

- **Zero heavy dependencies** — pure NumPy, so the math is transparent and
  auditable. Swap in PyTorch for production-scale training.
- **Amortized fairness** — the re-ranker tracks cumulative exposure *across
  requests*, so parity holds over time, not just per slate.
- **Bounded-variance debiasing** — IPS weights are clipped, trading a little
  bias for stable estimates on sparse logs.

## License

MIT — see [LICENSE](LICENSE).
