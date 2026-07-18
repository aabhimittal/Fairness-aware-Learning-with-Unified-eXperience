# FLUX

**Fairness-aware Learning with Unified eXperience** — a social media ranking
system that combines four techniques into one servable pipeline:

| Stage | Technique | Module |
|---|---|---|
| Relevance | Hyperbolic (Poincaré-ball) interest embeddings | `flux.hyperbolic` |
| Personalization | Quantum-inspired attention with context collapse | `flux.attention` |
| Debiasing | Clipped inverse propensity scoring | `flux.causal` |
| Fairness | Amortized exposure-parity re-ranking | `flux.fairness` |

The unified `FluxEngine` (`flux.engine`) chains all four in a single
`rank()` call; `feedback()` folds engagement back into the user state.

## Installation

```bash
pip install flux-ranking                 # NumPy core only
pip install "flux-ranking[torch]"        # + PyTorch training backend
pip install "flux-ranking[api]"          # + FastAPI serving layer
pip install "flux-ranking[eval]"         # + notebook/evaluation toolchain
```

## Where to go next

- [Concepts](concepts.md) — why hyperbolic geometry and quantum-style attention
- [Quickstart](quickstart.md) — rank a slate in ten lines
- [Evaluation](evaluation.md) — the `flux-eval` CLI and MovieLens results
- [Serving API](serving.md) — the FastAPI endpoint
- [Releasing](releasing.md) — how versions reach PyPI
