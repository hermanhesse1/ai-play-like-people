"""Nash equilibrium: the fully rational benchmark.

For each 2x2 game we find every equilibrium (pure, plus the completely mixed
one when it exists) and predict the average of the decider's probability of A
across them, as the plan specifies.

A pure equilibrium predicts 0 or 1, which gives an infinite log likelihood the
moment one person deviates. So for the likelihood we use a trembling version,
p = (1 - eps) * p_nash + eps / 2, with eps fitted on the training games. The
untrembled prediction (eps = 0) is still reported for MSE, which is how the
2025 paper scores Nash.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..games import own_coefficients, opp_coefficients, pure_equilibria, split
from .base import Model, binomial_nll


def equilibria(payoffs: np.ndarray, tol: float = 1e-12):
    """All Nash equilibria of each game.

    Returns (pure, mixed_p, has_mixed):
      pure      (N, 2, 2) bool, pure[n, i, j] = (i, j) is an equilibrium
      mixed_p   (N,) the decider's P(A) in the completely mixed equilibrium
      has_mixed (N,) bool, whether a completely mixed equilibrium exists

    In degenerate games (payoff ties) there can be a continuum of equilibria;
    we then keep only the pure ones, which are the endpoints.
    """
    U, V = split(payoffs)
    pure = pure_equilibria(U, V)
    a, b = own_coefficients(U)     # my gain from A: a*q + b
    c, d = opp_coefficients(V)     # their gain from A: c*p + d
    with np.errstate(divide="ignore", invalid="ignore"):
        q_star = np.where(np.abs(a) > tol, -b / a, np.nan)   # makes me indifferent
        p_star = np.where(np.abs(c) > tol, -d / c, np.nan)   # makes them indifferent
    has_mixed = (q_star > tol) & (q_star < 1 - tol) & (p_star > tol) & (p_star < 1 - tol)
    return pure, np.where(has_mixed, p_star, np.nan), has_mixed


def nash_prediction(payoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average decider P(A) over all equilibria, and the number of equilibria."""
    pure, mixed_p, has_mixed = equilibria(payoffs)
    # pure profiles where I play A are (0, 0) and (0, 1)
    n_pure = pure.sum(axis=(1, 2))
    sum_pa = pure[:, 0, :].sum(axis=1) + np.where(has_mixed, mixed_p, 0.0)
    count = n_pure + has_mixed
    return sum_pa / np.maximum(count, 1), count


class Nash(Model):
    name = "Nash"
    n_params = 1

    def __init__(self, eps: float = 0.0):
        self.eps = eps

    def fit(self, payoffs, n_a, n):
        base, _ = nash_prediction(payoffs)
        res = minimize_scalar(lambda e: binomial_nll((1 - e) * base + e / 2, n_a, n),
                              bounds=(1e-6, 1.0), method="bounded")
        self.eps = float(res.x)
        return self

    def predict(self, payoffs):
        base, _ = nash_prediction(payoffs)
        return (1 - self.eps) * base + self.eps / 2

    def predict_raw(self, payoffs):
        return nash_prediction(payoffs)[0]

    def params(self):
        return {"eps": self.eps}
