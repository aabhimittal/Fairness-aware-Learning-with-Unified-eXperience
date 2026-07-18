"""Quantum-inspired attention over user interests.

A user's interests are held as a superposition: complex amplitudes over
interest dimensions. Context (time of day, social signals, current session)
acts as a measurement operator that collapses the state into a probability
distribution used to weight content scores. Complex phases let interests
interfere -- constructively (related interests reinforce) or destructively
(conflicting interests suppress each other) -- which a plain softmax over
real weights cannot express.
"""

import numpy as np


class QuantumAttention:
    def __init__(self, n_interests: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        amp = rng.normal(size=n_interests) + 1j * rng.normal(size=n_interests)
        self.state = amp / np.linalg.norm(amp)  # unit-norm complex state |psi>

    @property
    def probabilities(self) -> np.ndarray:
        """Born rule: p_i = |amplitude_i|^2."""
        return np.abs(self.state) ** 2

    def evolve(self, engagement: np.ndarray, phase_rate: float = 0.5):
        """Unitary-style update from an engagement signal per interest.

        Positive engagement rotates phase forward and grows amplitude;
        the state is renormalized to stay a valid superposition.
        """
        engagement = np.asarray(engagement, dtype=float)
        rotation = np.exp(1j * phase_rate * engagement)
        amp = self.state * rotation * (1.0 + 0.1 * engagement)
        self.state = amp / np.linalg.norm(amp)

    def collapse(self, context: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Measure the state in a context basis; returns interest weights.

        `context` is a real non-negative relevance vector per interest
        (e.g. time-of-day or session affinity). The measurement reweights
        amplitudes before applying the Born rule, so the same user state
        yields different interest profiles in different contexts.
        """
        context = np.asarray(context, dtype=float)
        if np.any(context < 0):
            raise ValueError("context weights must be non-negative")
        measured = self.state * np.sqrt(context + 1e-12)
        p = np.abs(measured) ** 2
        p = p ** (1.0 / max(temperature, 1e-6))
        total = p.sum()
        if total < 1e-12:
            return np.full_like(p, 1.0 / len(p))
        return p / total

    def interference(self, i: int, j: int) -> float:
        """Signed interference between two interests in [-1, 1].

        cos(phase difference): +1 reinforcing, -1 conflicting.
        """
        return float(np.cos(np.angle(self.state[i]) - np.angle(self.state[j])))
