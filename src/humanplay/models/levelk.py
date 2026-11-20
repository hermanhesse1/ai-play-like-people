"""Limited strategic reasoning: level-k and Poisson cognitive hierarchy.

Level 0 plays uniformly at random. In level-k (Stahl and Wilson, 1994;
Nagel, 1995) a level k player responds to an opponent of level k - 1. In the
cognitive hierarchy model (Camerer, Ho and Chong, 2004) a level k player
responds to a Poisson(tau) mix of all lower levels, renormalised.

Responses are logit (noisy) best responses with precision lam, so the models
have a likelihood. Population P(A) is the level-weighted average.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from ..games import opp_gain, own_gain, sigmoid, split
from .base import Model, binomial_nll


def level_k_choices(payoffs, lam, K):
    """P(A) of each level 0..K, for me (P) and for the opponent (Q). Shape (K+1, N)."""
    U, V = split(payoffs)
    N = len(payoffs)
    P, Q = [np.full(N, 0.5)], [np.full(N, 0.5)]
    for k in range(1, K + 1):
        P.append(sigmoid(lam * own_gain(U, Q[k - 1])))
        Q.append(sigmoid(lam * opp_gain(V, P[k - 1])))
    return np.array(P), np.array(Q)


def ch_choices(payoffs, lam, tau, K):
    """P(A) of each CH level 0..K for me and the opponent, plus level weights."""
    U, V = split(payoffs)
    N = len(payoffs)
    f = poisson.pmf(np.arange(K + 1), tau)
    P, Q = [np.full(N, 0.5)], [np.full(N, 0.5)]
    for k in range(1, K + 1):
        w = f[:k] / f[:k].sum()
        q_belief = np.tensordot(w, np.array(Q), axes=1)
        p_belief = np.tensordot(w, np.array(P), axes=1)
        P.append(sigmoid(lam * own_gain(U, q_belief)))
        Q.append(sigmoid(lam * opp_gain(V, p_belief)))
    return np.array(P), np.array(Q), f / f.sum()


def _multistart(objective, starts, bounds):
    best = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best


class LevelK(Model):
    """Mixture of levels 0..K with free weights and one shared precision lam."""
    name = "Level-k"

    def __init__(self, K: int = 3, lam: float = 0.2, weights=None):
        self.K = K
        self.lam = lam
        self.weights = np.full(K + 1, 1 / (K + 1)) if weights is None else np.asarray(weights)
        self.n_params = K + 1          # K free weights + lam

    def _unpack(self, theta):
        lam = np.exp(theta[0])
        z = np.concatenate([[0.0], theta[1:]])          # level 0 logit fixed at 0
        w = np.exp(z - z.max())
        return lam, w / w.sum()

    def _predict(self, payoffs, lam, w):
        P, _ = level_k_choices(payoffs, lam, self.K)
        return w @ P

    def fit(self, payoffs, n_a, n):
        obj = lambda th: binomial_nll(self._predict(payoffs, *self._unpack(th)), n_a, n)
        starts = [np.r_[np.log(l), np.zeros(self.K)] for l in (0.05, 0.2, 1.0)]
        starts.append(np.r_[np.log(0.2), np.full(self.K, 1.0)])
        bounds = [(np.log(1e-4), np.log(20))] + [(-12, 12)] * self.K
        res = _multistart(obj, starts, bounds)
        self.lam, self.weights = self._unpack(res.x)
        return self

    def predict(self, payoffs):
        return self._predict(payoffs, self.lam, self.weights)

    def params(self):
        out = {"lambda": self.lam}
        out.update({f"w{k}": w for k, w in enumerate(self.weights)})
        return out


class CognitiveHierarchy(Model):
    """Poisson cognitive hierarchy with logit responses: parameters tau and lam."""
    name = "Cognitive hierarchy"
    n_params = 2

    def __init__(self, tau: float = 1.5, lam: float = 0.2, K: int = 7):
        self.tau, self.lam, self.K = tau, lam, K

    def _predict(self, payoffs, lam, tau):
        P, _, f = ch_choices(payoffs, lam, tau, self.K)
        return f @ P

    def fit(self, payoffs, n_a, n):
        obj = lambda th: binomial_nll(self._predict(payoffs, np.exp(th[0]), np.exp(th[1])), n_a, n)
        starts = [np.log([l, t]) for l in (0.05, 0.3) for t in (0.5, 1.5, 4.0)]
        bounds = [(np.log(1e-4), np.log(20)), (np.log(1e-3), np.log(10))]
        res = _multistart(obj, starts, bounds)
        self.lam, self.tau = np.exp(res.x)
        return self

    def predict(self, payoffs):
        return self._predict(payoffs, self.lam, self.tau)

    def params(self):
        return {"tau": self.tau, "lambda": self.lam}
