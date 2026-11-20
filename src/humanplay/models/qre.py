"""Logit quantal response equilibrium (McKelvey and Palfrey, 1995).

Both players make noisy best responses, p = sigmoid(lam * my_gain(q)) and
q = sigmoid(lam * their_gain(p)), and each is right about the other's noise.

Substituting q into p gives a one dimensional equation. We solve it on the
logit scale, x = logit(p), which stays well behaved when p is near 0 or 1:

    h(x, lam) = x - lam * (a * sigmoid(lam * (c * sigmoid(x) + d)) + b) = 0

When lam is large, coordination games have three solutions. The standard
choice is the principal branch: the one connected to uniform play at lam = 0.
We follow it by continuation:

* small steps in lam (at most DLAM), each starting from a tangent prediction
  dx/dlam = -h_lam / h_x, then corrected by Newton's method;
* only stable solutions (h_x > 0) are accepted. The principal branch starts
  with h_x = 1 and keeps h_x > 0 unless it reaches a fold;
* at a fold the branch turns back in lam and the root being followed
  disappears. A 2x2 game has at most three solutions, so exactly one stable
  solution remains, and the search moves to it. Beyond the fold that is the
  only solution, so it is where the principal branch continues.

Coarse steps without these checks can jump to a different solution near the
point where new solutions appear; tests/test_models.py has such games.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..games import own_coefficients, opp_coefficients, sigmoid, split
from .base import Model, binomial_nll

DLAM = 0.002


def _h(x, lam, a, b, c, d):
    """h, dh/dx and dh/dlam at (x, lam)."""
    p = sigmoid(x)
    u = c * p + d
    q = sigmoid(lam * u)
    h = x - lam * (a * q + b)
    hx = 1.0 - lam * a * q * (1 - q) * lam * c * p * (1 - p)
    hlam = -(a * q + b) - lam * a * q * (1 - q) * u
    return h, hx, hlam


def _stable_root_near(x0, lam, a, b, c, d):
    """The stable solution (h crosses from - to +) closest to x0, for one game."""
    bound = lam * (abs(a) + abs(b)) + 1.0          # every solution lies in [-bound, bound]
    xs = np.linspace(-bound, bound, 20001)
    f = lambda x: _h(np.asarray(x, dtype=float), lam, a, b, c, d)[0]
    hs = f(xs)
    up = np.where((hs[:-1] < 0) & (hs[1:] >= 0))[0]
    roots = [brentq(lambda x: float(f(x)), xs[i], xs[i + 1], xtol=1e-13) for i in up]
    return min(roots, key=lambda r: abs(r - x0))


def _correct(x_pred, lam, a, b, c, d, iters=8, tol=1e-9):
    x = x_pred.copy()
    for _ in range(iters):
        h, hx, _ = _h(x, lam, a, b, c, d)
        x = x - np.clip(h / np.where(np.abs(hx) < 1e-9, 1e-9, hx), -1.0, 1.0)
    h, hx, _ = _h(x, lam, a, b, c, d)
    bad = np.where(~((np.abs(h) < tol) & (hx > 0)))[0]
    for i in bad:
        x[i] = _stable_root_near(x_pred[i], lam, a[i], b[i], c[i], d[i])
    return x


def _follow(x, lam0, lam1, a, b, c, d, dlam=DLAM):
    """Move the principal-branch solutions x from lam0 to lam1."""
    if lam1 <= lam0:
        return x
    grid = np.linspace(lam0, lam1, int(np.ceil((lam1 - lam0) / dlam)) + 1)
    for l0, l1 in zip(grid[:-1], grid[1:]):
        _, hx, hlam = _h(x, l0, a, b, c, d)
        slope = -hlam / np.where(np.abs(hx) < 1e-3, 1e-3, hx)
        x = _correct(x + np.clip(slope * (l1 - l0), -1.0, 1.0), l1, a, b, c, d)
    return x


def qre_path(payoffs: np.ndarray, lambdas: np.ndarray):
    """Principal-branch QRE for each game along an increasing grid of lam.

    Returns P (L, N) and Q (L, N): decider and opponent P(A) at each lam.
    """
    U, V = split(payoffs)
    a, b = own_coefficients(U)
    c, d = opp_coefficients(V)
    lambdas = np.asarray(lambdas, dtype=float)
    if np.any(np.diff(lambdas) < 0) or np.any(lambdas < 0):
        raise ValueError("lambdas must be non-negative and increasing")
    x = np.zeros(len(payoffs))
    prev = 0.0
    P, Q = [], []
    for lam in lambdas:
        x = _follow(x, prev, lam, a, b, c, d)
        prev = lam
        p = sigmoid(x)
        P.append(p)
        Q.append(sigmoid(lam * (c * p + d)))
    return np.array(P), np.array(Q)


def qre(payoffs: np.ndarray, lam: float):
    """Principal-branch QRE at a single lam."""
    P, Q = qre_path(payoffs, np.array([lam]))
    return P[0], Q[0]


class QRE(Model):
    name = "QRE"
    n_params = 1
    GRID = np.geomspace(1e-3, 5.0, 240)

    def __init__(self, lam: float = 0.1):
        self.lam = lam

    def fit(self, payoffs, n_a, n):
        P, _ = qre_path(payoffs, self.GRID)
        losses = np.array([binomial_nll(p, n_a, n) for p in P])
        i = int(np.argmin(losses))
        # refine between the neighbouring grid points with a short golden search
        lo, hi = self.GRID[max(i - 1, 0)], self.GRID[min(i + 1, len(self.GRID) - 1)]
        best_lam, best_loss = self.GRID[i], losses[i]
        g = (np.sqrt(5) - 1) / 2
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f = lambda lam: binomial_nll(qre(payoffs, lam)[0], n_a, n)
        f1, f2 = f(x1), f(x2)
        for _ in range(12):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = f(x2)
        for lam, loss in ((x1, f1), (x2, f2)):
            if loss < best_loss:
                best_lam, best_loss = lam, loss
        self.lam = float(best_lam)
        self.path_ = (self.GRID, losses)           # kept for plotting the fit
        return self

    def predict(self, payoffs):
        return qre(payoffs, self.lam)[0]

    def params(self):
        return {"lambda": self.lam}
