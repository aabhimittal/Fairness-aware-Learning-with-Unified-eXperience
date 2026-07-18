# Serving API

`flux.serving.create_app(engine, catalog)` wraps a `FluxEngine` and a fixed
catalog in a FastAPI app. Requires the api extra:

```bash
pip install "flux-ranking[api]"
uvicorn examples.serve_api:app --reload
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + catalog size |
| POST | `/rank` | Rank the catalog for a user point + context |
| POST | `/feedback` | Fold engagement into the quantum interest state |
| GET | `/fairness` | Cumulative exposure shares vs. targets |

## Examples

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/rank \
  -H 'content-type: application/json' \
  -d '{"user_point": [0.1, 0.0, 0.05, -0.02],
       "context": [1.0, 2.0, 0.3], "k": 5}'

curl -s -X POST localhost:8000/feedback \
  -H 'content-type: application/json' \
  -d '{"engagement": [0.2, 1.0, -0.3]}'

curl -s localhost:8000/fairness
```

`/rank` returns the served slate in order; `/fairness` shows how cumulative
exposure tracks the configured targets as slates are served — the amortized
fairness guarantee in action.

!!! note
    The example server (`examples/serve_api.py`) holds one engine in process
    memory: single-user, demo-scale. Production use would shard engines per
    user and persist attention state externally.
