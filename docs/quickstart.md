# Quickstart

## Serve a fair slate (NumPy core)

```python
import numpy as np
from flux import FluxEngine, Content

engine = FluxEngine(
    n_interests=3,
    target_shares={"established": 0.55, "emerging": 0.45},
    fairness_weight=2.0,
)

catalog = [
    Content(id="post-0", creator_group="established",
            embedding=np.array([0.05, 0.0, 0.0, 0.0]),
            interest_affinity=np.array([0.7, 0.2, 0.1]),
            logged_clicks=8.0, logged_rank=0),
    # ... more Content ...
]

user_point = np.array([0.1, 0.05, -0.08, 0.02])
context = np.array([1.0, 2.0, 0.3])       # e.g. fitness-heavy morning

slate = engine.rank(user_point, catalog, context, k=10)
engine.feedback(np.array([0.2, 1.0, -0.3]))  # engagement evolves the state
```

## Train on real data (torch extra)

```python
import torch
from flux.datasets import load_ml100k         # or load_ml1m
from flux.torch_backend import HyperbolicInterestModel, TrainConfig, train

ml = load_ml100k()                            # cached ~5 MB download
model = HyperbolicInterestModel(ml.n_users, ml.n_items, dim=16)
train(model,
      torch.from_numpy(ml.train[:, 0]), torch.from_numpy(ml.train[:, 1]),
      n_items=ml.n_items, config=TrainConfig(epochs=40))

top10 = model.rank_items(user=0, k=10)
```

## Evaluate from the command line

```bash
flux-eval --dataset ml-100k --epochs 40 --output results.json
flux-eval --dataset ml-1m --epochs 20        # larger, ~1M ratings
```

## Run the demo

```bash
python examples/demo.py
```
