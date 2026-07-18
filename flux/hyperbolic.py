"""Poincare-ball hyperbolic geometry for hierarchical interest embeddings.

General interests (e.g. "dogs") live near the origin; specific ones
(e.g. "golden retrievers") sit near the boundary, where hyperbolic space
grows exponentially -- matching the branching factor of interest taxonomies.
"""

import numpy as np

_EPS = 1e-9
_MAX_NORM = 1.0 - 1e-5


class PoincareBall:
    """Operations on the Poincare ball of curvature -c."""

    def __init__(self, c: float = 1.0):
        if c <= 0:
            raise ValueError("curvature parameter c must be positive")
        self.c = c
        self.sqrt_c = np.sqrt(c)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Clip points back inside the ball (numerical safety)."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = _MAX_NORM / self.sqrt_c
        factor = np.where(norm > max_norm, max_norm / (norm + _EPS), 1.0)
        return x * factor

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mobius addition x (+) y — the hyperbolic analogue of vector addition."""
        c = self.c
        xy = np.sum(x * y, axis=-1, keepdims=True)
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        y2 = np.sum(y * y, axis=-1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c**2 * x2 * y2
        return self.project(num / (den + _EPS))

    def dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Geodesic distance between points on the ball."""
        diff = self.mobius_add(-x, y)
        norm = np.linalg.norm(diff, axis=-1)
        return (2.0 / self.sqrt_c) * np.arctanh(
            np.clip(self.sqrt_c * norm, 0.0, _MAX_NORM)
        )

    def expmap0(self, v: np.ndarray) -> np.ndarray:
        """Map a tangent vector at the origin onto the ball."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        norm = np.maximum(norm, _EPS)
        return self.project(np.tanh(self.sqrt_c * norm) * v / (self.sqrt_c * norm))

    def logmap0(self, x: np.ndarray) -> np.ndarray:
        """Map a ball point back to the tangent space at the origin."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.maximum(norm, _EPS)
        scaled = np.clip(self.sqrt_c * norm, _EPS, _MAX_NORM)
        return np.arctanh(scaled) * x / (self.sqrt_c * norm)

    def hierarchy_level(self, x: np.ndarray) -> np.ndarray:
        """Distance from origin: ~0 = broad interest, large = niche interest."""
        return self.dist(np.zeros_like(x), x)


class HyperbolicEmbedding:
    """Learnable embeddings on the Poincare ball, trained with Riemannian SGD."""

    def __init__(self, n_items: int, dim: int = 16, c: float = 1.0, seed: int = 0):
        self.ball = PoincareBall(c)
        rng = np.random.default_rng(seed)
        # small init near the origin, where optimization is well-conditioned
        self.weights = self.ball.project(rng.normal(0, 1e-2, size=(n_items, dim)))

    def dist(self, i: int, j: int) -> float:
        return float(self.ball.dist(self.weights[i], self.weights[j]))

    def riemannian_update(self, idx: int, euclidean_grad: np.ndarray, lr: float = 0.1):
        """RSGD step: rescale the Euclidean gradient by the inverse metric."""
        x = self.weights[idx]
        conformal = (1 - self.ball.c * np.sum(x * x)) ** 2 / 4.0
        self.weights[idx] = self.ball.project(x - lr * conformal * euclidean_grad)

    def train_step(self, anchor: int, positive: int, negative: int, lr: float = 0.1,
                   margin: float = 1.0) -> float:
        """One triplet step: pull anchor toward positive, push from negative.

        Uses a finite-difference gradient of the margin loss so the module
        stays dependency-free; returns the loss value.
        """
        def loss() -> float:
            d_pos = self.dist(anchor, positive)
            d_neg = self.dist(anchor, negative)
            return max(0.0, margin + d_pos - d_neg)

        base = loss()
        if base == 0.0:
            return 0.0
        grad = np.zeros_like(self.weights[anchor])
        h = 1e-5
        for k in range(len(grad)):
            self.weights[anchor][k] += h
            grad[k] = (loss() - base) / h
            self.weights[anchor][k] -= h
        self.riemannian_update(anchor, grad, lr)
        return base
