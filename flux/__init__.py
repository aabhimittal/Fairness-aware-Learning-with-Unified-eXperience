"""FLUX: Fairness-aware Learning with Unified eXperience.

A social-media ranking system combining:
- Hyperbolic (Poincare ball) interest embeddings for hierarchical interests
- Quantum-inspired attention (superposed interests collapsing on context)
- Causal debiasing via inverse propensity scoring
- Fairness-aware exposure re-ranking
"""

from .hyperbolic import PoincareBall, HyperbolicEmbedding
from .attention import QuantumAttention
from .causal import IPSDebiaser
from .fairness import FairReranker
from .engine import FluxEngine, Content
from .datasets import MovieLens, load_ml100k, parse_ml100k

__version__ = "0.2.0"
__all__ = [
    "MovieLens",
    "load_ml100k",
    "parse_ml100k",
    "PoincareBall",
    "HyperbolicEmbedding",
    "QuantumAttention",
    "IPSDebiaser",
    "FairReranker",
    "FluxEngine",
    "Content",
]
