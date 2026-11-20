"""Representation of 2x2 game instances and basic game theory helpers.

Every game instance is stored from the point of view of the player who decides
(the paper adapted every presentation to a row player perspective). One
instance is a vector of 8 payoffs:

    index:   0     1     2     3     4     5     6     7
             U_AA  U_AB  U_BA  U_BB  V_AA  V_AB  V_BA  V_BB

    U[i, j] = my payoff when I play i and the opponent plays j
    V[i, j] = the opponent's payoff when I play i and the opponent plays j

so U and V are indexed the same way (my action first). The target is the share
of deciding players who chose A.

All functions are vectorised over a leading axis of N game instances.
"""

from __future__ import annotations

import numpy as np

PAYOFF_COLUMNS = ["u_aa", "u_ab", "u_ba", "u_bb", "v_aa", "v_ab", "v_ba", "v_bb"]

# Labels for the breakdown by strategic structure, from the decider's view.
CATEGORIES = ("own_dominant", "opp_dominant", "no_dominant")


def split(payoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 8) payoff vectors -> U, V, each of shape (N, 2, 2)."""
    P = np.asarray(payoffs, dtype=float)
    if P.ndim == 1:
        P = P[None, :]
    return P[:, :4].reshape(-1, 2, 2), P[:, 4:].reshape(-1, 2, 2)


def join(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Inverse of split."""
    return np.concatenate([U.reshape(-1, 4), V.reshape(-1, 4)], axis=1)


def flip_perspective(payoffs: np.ndarray) -> np.ndarray:
    """The same game seen by the opponent.

    The opponent's own payoff for (their action j, my action i) is V[i, j], so
    their U is V transposed, and their V is my U transposed.
    """
    U, V = split(payoffs)
    return join(V.transpose(0, 2, 1), U.transpose(0, 2, 1))


def own_coefficients(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gain from A over B for me is a * q + b, where q = P(opponent plays A)."""
    a = (U[:, 0, 0] - U[:, 1, 0]) - (U[:, 0, 1] - U[:, 1, 1])
    b = U[:, 0, 1] - U[:, 1, 1]
    return a, b


def opp_coefficients(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gain from A over B for the opponent is c * p + d, where p = P(I play A)."""
    c = (V[:, 0, 0] - V[:, 0, 1]) - (V[:, 1, 0] - V[:, 1, 1])
    d = V[:, 1, 0] - V[:, 1, 1]
    return c, d


def own_gain(U: np.ndarray, q: np.ndarray | float) -> np.ndarray:
    """Expected payoff of A minus B for me, if the opponent plays A with prob q."""
    a, b = own_coefficients(U)
    return a * q + b


def opp_gain(V: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    """Expected payoff of A minus B for the opponent, if I play A with prob p."""
    c, d = opp_coefficients(V)
    return c * p + d


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function (the 2-action softmax)."""
    out = np.empty_like(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def dominance(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strict dominance for each side.

    Returns two int arrays (own, opp) with values +1 if A strictly dominates B,
    -1 if B strictly dominates A, and 0 if neither action is strictly dominant.
    """
    own = np.where((U[:, 0, 0] > U[:, 1, 0]) & (U[:, 0, 1] > U[:, 1, 1]), 1,
                   np.where((U[:, 0, 0] < U[:, 1, 0]) & (U[:, 0, 1] < U[:, 1, 1]), -1, 0))
    opp = np.where((V[:, 0, 0] > V[:, 0, 1]) & (V[:, 1, 0] > V[:, 1, 1]), 1,
                   np.where((V[:, 0, 0] < V[:, 0, 1]) & (V[:, 1, 0] < V[:, 1, 1]), -1, 0))
    return own, opp


def pure_equilibria(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Boolean array (N, 2, 2): is the profile (my i, their j) a pure Nash equilibrium?"""
    own_best = U >= U[:, ::-1, :]          # i is a best response to j
    opp_best = V >= V[:, :, ::-1]          # j is a best response to i
    return own_best & opp_best


def categorise(payoffs: np.ndarray) -> np.ndarray:
    """Strategic structure from the decider's point of view.

    own_dominant : I have a strictly dominant action (zero steps of reasoning
                   about the opponent needed).
    opp_dominant : I don't, but the opponent does, so the game is solvable if I
                   reason one step about them. This is where level-k depth shows.
    no_dominant  : neither player has a dominant action. In this dataset every
                   game has a pure equilibrium, so these are mostly games with two
                   pure equilibria (coordination type).

    Note that the dataset contains no games with only mixed equilibria: the
    authors dropped games without a pure-strategy equilibrium.
    """
    U, V = split(payoffs)
    own, opp = dominance(U, V)
    cat = np.where(own != 0, "own_dominant", np.where(opp != 0, "opp_dominant", "no_dominant"))
    return cat.astype(object)
