"""Optional PyTorch training backend for FLUX.

Requires `torch` (see requirements-torch.txt); the NumPy core package
works without it.
"""

try:
    import torch  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "flux.torch_backend requires PyTorch: pip install -r requirements-torch.txt"
    ) from e

from .manifold import PoincareManifold
from .model import HyperbolicInterestModel
from .train import TrainConfig, train, make_synthetic_interactions

__all__ = [
    "PoincareManifold",
    "HyperbolicInterestModel",
    "TrainConfig",
    "train",
    "make_synthetic_interactions",
]
