"""FastAPI serving layer around FluxEngine (requires the api extra).

    from flux.serving import create_app
    app = create_app(engine, catalog)   # catalog: list[Content]

Run: uvicorn examples.serve_api:app --reload  (see examples/serve_api.py)
"""

import numpy as np

from .engine import Content, FluxEngine

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'flux.serving requires the api extra: pip install "flux-ranking[api]"'
    ) from e


class RankRequest(BaseModel):
    user_point: list[float]  # position on the Poincare ball
    context: list[float]     # non-negative per-interest context weights
    k: int | None = None


class FeedbackRequest(BaseModel):
    engagement: list[float]  # signed per-interest engagement signal


def create_app(engine: FluxEngine, catalog: list[Content]) -> "FastAPI":
    """Build a FastAPI app serving one FluxEngine over a fixed catalog."""
    app = FastAPI(title="FLUX ranking API",
                  description="Fairness-aware feed ranking")

    @app.get("/health")
    def health():
        return {"status": "ok", "catalog_size": len(catalog)}

    @app.post("/rank")
    def rank(req: RankRequest):
        n = len(engine.attention.state)
        if len(req.context) != n:
            raise HTTPException(422, f"context must have {n} entries")
        slate = engine.rank(np.array(req.user_point), catalog,
                            context=np.array(req.context), k=req.k)
        return {"slate": [
            {"id": c.id, "creator_group": c.creator_group} for c in slate]}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        n = len(engine.attention.state)
        if len(req.engagement) != n:
            raise HTTPException(422, f"engagement must have {n} entries")
        engine.feedback(np.array(req.engagement))
        return {"interest_probabilities":
                engine.attention.probabilities.round(4).tolist()}

    @app.get("/fairness")
    def fairness():
        return {"exposure_shares": engine.reranker.exposure_shares(),
                "targets": engine.reranker.target_shares}

    return app
