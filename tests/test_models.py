"""Checks on textbook games, fixed-point accuracy, and parameter recovery."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanplay.data import both_perspectives, generate_games, simulate  # noqa: E402
from humanplay.evaluate import bootstrap, game_folds, noise_floor  # noqa: E402
from humanplay.games import (categorise, flip_perspective, opp_gain, own_gain,  # noqa: E402
                             pure_equilibria, sigmoid, split)
from humanplay.models import QRE, CognitiveHierarchy, LevelK, Nash, Random  # noqa: E402
from humanplay.models.levelk import ch_choices, level_k_choices  # noqa: E402
from humanplay.models.nash import nash_prediction  # noqa: E402
from humanplay.models.qre import qre, qre_path  # noqa: E402

# Layout: U_AA U_AB U_BA U_BB | V_AA V_AB V_BA V_BB (my action first in both)
PRISONERS = np.array([3, 0, 5, 1, 3, 5, 0, 1.])      # A = cooperate
SEXES = np.array([3, 0, 0, 2, 2, 0, 0, 3.])
PENNIES = np.array([1, 0, 0, 1, 0, 1, 1, 0.])
# Only the opponent has a dominant action (A); my best reply to A is B.
OPP_DOM = np.array([1, 4, 3, 2, 5, 1, 6, 2.])


@pytest.fixture(scope="module")
def games():
    payoffs, ids = both_perspectives(generate_games(500, np.random.default_rng(1)))
    return payoffs, ids


# ---------------------------------------------------------------- representation

def test_flip_is_an_involution(games):
    payoffs, _ = games
    assert np.allclose(flip_perspective(flip_perspective(payoffs)), payoffs)


def test_symmetric_game_looks_the_same_from_both_sides():
    assert np.allclose(flip_perspective(PRISONERS[None])[0], PRISONERS)


def test_categories():
    cats = categorise(np.stack([PRISONERS, OPP_DOM, SEXES]))
    assert list(cats) == ["own_dominant", "opp_dominant", "no_dominant"]


def test_generated_games_all_have_pure_equilibria(games):
    payoffs, _ = games
    U, V = split(payoffs)
    assert pure_equilibria(U, V).any(axis=(1, 2)).all()
    assert payoffs.min() >= 1 and payoffs.max() <= 50


# ---------------------------------------------------------------- Nash

def test_nash_textbook_games():
    p, count = nash_prediction(np.stack([PRISONERS, SEXES, PENNIES, OPP_DOM]))
    assert p[0] == 0 and count[0] == 1                 # defect
    assert p[1] == pytest.approx((1 + 0 + 0.6) / 3)    # two pure + mixed at p = 0.6
    assert count[1] == 3
    assert p[2] == pytest.approx(0.5) and count[2] == 1
    assert p[3] == 0                                   # opponent plays A, I reply B


def test_nash_tremble_is_recovered(games):
    payoffs, ids = games
    base, _ = nash_prediction(payoffs)
    p = 0.8 * base + 0.1
    rng = np.random.default_rng(0)
    n = np.full(len(p), 400)
    model = Nash().fit(payoffs, rng.binomial(n, p), n)
    assert model.eps == pytest.approx(0.2, abs=0.02)


# ---------------------------------------------------------------- QRE

def test_qre_is_uniform_at_zero_and_nash_in_dominance_games():
    G = np.stack([PRISONERS, PENNIES])
    assert np.allclose(qre(G, 0.0)[0], 0.5)
    p, _ = qre(G, 20.0)
    assert p[0] < 1e-6 and p[1] == pytest.approx(0.5)


def test_qre_solves_its_fixed_point_on_every_game(games):
    payoffs, _ = games
    U, V = split(payoffs)
    lambdas = np.array([0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
    P, Q = qre_path(payoffs, lambdas)
    for lam, p, q in zip(lambdas, P, Q):
        assert np.max(np.abs(p - sigmoid(lam * own_gain(U, q)))) < 1e-8
        assert np.max(np.abs(q - sigmoid(lam * opp_gain(V, p)))) < 1e-8


def test_qre_single_and_path_agree(games):
    payoffs, _ = games
    P, _ = qre_path(payoffs, QRE.GRID)
    i = np.searchsorted(QRE.GRID, 0.1)
    assert np.allclose(qre(payoffs, QRE.GRID[i])[0], P[i], atol=1e-6)


# Games where new QRE solutions appear partway along the path from lam = 0, so
# coarse continuation can jump onto the wrong one. FOLD's principal branch
# really does turn back near lam = 1.07: above it only the low solution exists.
BRANCHY = np.array([[2, 19, 18, 5, 9, 44, 40, 9],
                    [20, 17, 3, 36, 18, 2, 8, 26],
                    [39, 1, 2, 7, 20, 12, 2, 33],
                    [43, 7, 25, 38, 31, 4, 11, 26]], dtype=float)
FOLD = np.array([[4, 44, 12, 2, 4, 3, 12, 5]], dtype=float)


def _fine_reference(P, lam, dlam=2e-5):
    """Tiny plain steps: slow but cannot skip between solutions."""
    from humanplay.models.qre import _h
    from humanplay.games import own_coefficients, opp_coefficients
    U, V = split(P)
    a, b = own_coefficients(U)
    c, d = opp_coefficients(V)
    x = np.zeros(len(P))
    for s in np.linspace(dlam, lam, int(round(lam / dlam))):
        for _ in range(3):
            h, hx, _ = _h(x, s, a, b, c, d)
            x = x - h / hx
    return sigmoid(x)


@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_qre_follows_the_principal_branch(lam):
    G = np.concatenate([BRANCHY, flip_perspective(BRANCHY)])
    p, q = qre(G, lam)
    assert np.allclose(p, _fine_reference(G, lam), atol=1e-6)
    # each seat's answer must agree with the other seat's view of the same game
    assert np.allclose(q[:4], p[4:], atol=1e-9) and np.allclose(q[4:], p[:4], atol=1e-9)


def test_qre_crosses_a_fold():
    assert qre(FOLD, 1.0)[0][0] > 0.99          # still on the upper solution
    assert qre(FOLD, 1.2)[0][0] < 0.01          # branch has turned; only the low one is left


def test_qre_perspectives_are_consistent(games):
    """The opponent's QRE probability from my side equals my probability from theirs."""
    payoffs, _ = games
    p, q = qre(payoffs, 0.2)
    p_flipped, _ = qre(flip_perspective(payoffs), 0.2)
    assert np.allclose(q, p_flipped, atol=1e-6)


# ---------------------------------------------------------------- level-k, CH

def test_level_k_reasoning_steps():
    P, _ = level_k_choices(OPP_DOM[None], lam=10.0, K=2)
    assert P[0, 0] == 0.5
    assert P[2, 0] < 0.01        # level 2 expects the opponent's dominant A and replies B


def test_ch_with_tiny_tau_is_random(games):
    payoffs, _ = games
    P, _, f = ch_choices(payoffs, lam=0.2, tau=1e-4, K=5)
    assert np.allclose(f @ P, 0.5, atol=1e-3)


# ---------------------------------------------------------------- recovery

@pytest.mark.parametrize("truth,check", [
    (QRE(lam=0.08), lambda m: abs(m.lam / 0.08 - 1) < 0.05),
    (CognitiveHierarchy(tau=1.5, lam=0.15),
     lambda m: abs(m.tau / 1.5 - 1) < 0.15 and abs(m.lam / 0.15 - 1) < 0.10),
    (LevelK(K=3, lam=0.15, weights=[0.2, 0.5, 0.3, 0.0]),
     lambda m: abs(m.lam / 0.15 - 1) < 0.10 and abs(m.weights[1] - 0.5) < 0.08),
])
def test_parameter_recovery(truth, check):
    data = simulate(truth, n_games=1208, mean_n=40, seed=3)
    fitted = type(truth)().fit(data.payoffs, data.n_a, data.n)
    assert check(fitted), repr(fitted)


# ---------------------------------------------------------------- evaluation

def test_folds_keep_both_perspectives_together():
    data = simulate(Random(), n_games=300, seed=0)
    folds = game_folds(data, k=5)
    for g in np.unique(data.game_id):
        assert len(set(folds[data.game_id == g])) == 1


def test_noise_floor_matches_binomial_variance():
    data = simulate(QRE(lam=0.1), n_games=1208, mean_n=40, seed=1)
    true_p = QRE(lam=0.1).predict(data.payoffs)
    expected = np.mean(true_p * (1 - true_p) / data.n)
    assert noise_floor(data)["mse_floor"] == pytest.approx(expected, rel=0.08)


def test_bootstrap_intervals_cover_point_estimates():
    data = simulate(QRE(lam=0.1), n_games=400, seed=2)
    preds = {"Random": Random().predict(data.payoffs), "QRE": QRE(lam=0.1).predict(data.payoffs)}
    table, best = bootstrap(preds, data, B=300)
    assert best == "QRE"
    assert ((table.nll_lo <= table.nll) & (table.nll <= table.nll_hi)).all()
