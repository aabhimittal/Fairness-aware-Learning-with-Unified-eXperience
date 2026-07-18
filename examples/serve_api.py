"""Runnable FLUX API server with a small demo catalog.

    pip install "flux-ranking[api]"
    uvicorn examples.serve_api:app --reload

Then:
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/rank -H 'content-type: application/json' \
      -d '{"user_point": [0.1, 0.0, 0.05, -0.02], "context": [1.0, 2.0, 0.3], "k": 5}'
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from flux import Content, FluxEngine
from flux.serving import create_app

rng = np.random.default_rng(7)
N_INTERESTS = 3

engine = FluxEngine(
    n_interests=N_INTERESTS,
    target_shares={"established": 0.55, "emerging": 0.45},
    fairness_weight=2.0,
    seed=7,
)

catalog = [
    Content(
        id=f"post-{i}",
        creator_group="emerging" if i % 2 else "established",
        embedding=rng.normal(0, 0.15, size=4),
        interest_affinity=rng.dirichlet(np.ones(N_INTERESTS)),
        logged_clicks=float(rng.integers(0, 8)),
        logged_rank=int(rng.integers(0, 10)),
    )
    for i in range(20)
]

app = create_app(engine, catalog)
