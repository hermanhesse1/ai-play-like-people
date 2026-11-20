"""Common interface for every artificial player.

A model maps payoff vectors (decider's view) to the probability that the
decider plays A. Fitting maximises the binomial likelihood of the observed
counts, which is the same objective for every model, so scores are comparable.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def binomial_nll(p: np.ndarray, n_a: np.ndarray, n: np.ndarray) -> float:
    """Average negative log likelihood per decision (natural log)."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(n_a * np.log(p) + (n - n_a) * np.log(1 - p)).sum() / n.sum())


class Model:
    name = "model"
    n_params = 0

    def fit(self, payoffs: np.ndarray, n_a: np.ndarray, n: np.ndarray) -> "Model":
        return self

    def predict(self, payoffs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v:.4g}" for k, v in self.params().items())
        return f"{self.name}({inner})"


class Random(Model):
    """Everyone flips a coin. The floor every model must beat."""
    name = "Random"

    def predict(self, payoffs):
        return np.full(len(payoffs), 0.5)
