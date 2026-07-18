"""Differentiable Poincare-ball operations (autograd replaces the
finite-difference gradients of the NumPy core)."""

import torch

_EPS = 1e-9
_MAX_NORM = 1.0 - 1e-5


class PoincareManifold:
    """Poincare ball of curvature -c, with torch ops that support autograd."""

    def __init__(self, c: float = 1.0):
        if c <= 0:
            raise ValueError("curvature parameter c must be positive")
        self.c = c
        self.sqrt_c = c ** 0.5

    def project(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        max_norm = _MAX_NORM / self.sqrt_c
        return torch.where(norm > max_norm, x * (max_norm / norm), x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        xy = (x * y).sum(-1, keepdim=True)
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c**2 * x2 * y2
        return self.project(num / den.clamp_min(_EPS))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff_norm = self.mobius_add(-x, y).norm(dim=-1)
        return (2.0 / self.sqrt_c) * torch.atanh(
            (self.sqrt_c * diff_norm).clamp(0.0, _MAX_NORM)
        )

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return self.project(torch.tanh(self.sqrt_c * norm) * v / (self.sqrt_c * norm))

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        scaled = (self.sqrt_c * norm).clamp(_EPS, _MAX_NORM)
        return torch.atanh(scaled) * x / (self.sqrt_c * norm)

    def egrad2rgrad(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Rescale a Euclidean gradient by the inverse conformal metric."""
        factor = ((1 - self.c * (x * x).sum(-1, keepdim=True)) ** 2) / 4.0
        return factor * grad
