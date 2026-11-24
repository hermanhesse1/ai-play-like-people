"""Checks on the stage 2 neural network.

Skipped automatically if PyTorch is not installed, so the stage 1 tests still
run on their own.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("torch")

from humanplay.data import ProsocialMix, both_perspectives, generate_games, simulate  # noqa: E402
from humanplay.evaluate import game_folds, metrics  # noqa: E402
from humanplay.games import flip_perspective  # noqa: E402
from humanplay.models import CognitiveHierarchy, all_models  # noqa: E402
from humanplay.models.neural import NeuralNet, game_keys  # noqa: E402

CH_TRUTH = CognitiveHierarchy(tau=1.5, lam=0.15)
FAST = dict(n_ensemble=2, max_epochs=1500, patience=100)


def split_by_game(data, seed=0):
    """One train/test split by game (fold 0 of 5 is the test set)."""
    test = game_folds(data, k=5, seed=seed) == 0
    return data.subset(~test), data.subset(test)


def test_game_keys_pair_up_the_two_perspectives():
    payoffs, ids = both_perspectives(generate_games(200, np.random.default_rng(0)))
    keys = game_keys(payoffs)
    for g in np.unique(ids):
        assert len(set(keys[ids == g])) == 1                 # same game, same key
    assert len(np.unique(keys)) == len(np.unique(ids))       # different games differ
    assert np.array_equal(game_keys(flip_perspective(payoffs)), keys)


def test_early_stopping_set_holds_whole_games():
    payoffs, ids = both_perspectives(generate_games(300, np.random.default_rng(1)))
    train = NeuralNet()._split(payoffs, np.random.default_rng(0))
    for g in np.unique(ids):
        assert len(set(train[ids == g])) == 1
    assert 0.15 < 1 - train.mean() < 0.25


def test_predictions_are_probabilities_and_reproducible():
    data = simulate(CH_TRUTH, n_games=300, seed=0)
    a = NeuralNet(seed=7, **FAST).fit(data.payoffs, data.n_a, data.n).predict(data.payoffs)
    b = NeuralNet(seed=7, **FAST).fit(data.payoffs, data.n_a, data.n).predict(data.payoffs)
    assert a.shape == (len(data),)
    assert np.all((a > 0) & (a < 1))
    assert np.allclose(a, b)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        NeuralNet().predict(np.ones((2, 8)))


def test_network_learns_a_classic_model_nearly_as_well_as_the_truth():
    """Data from cognitive hierarchy: the network has no theory, but should get close."""
    data = simulate(CH_TRUTH, n_games=1208, seed=4)
    train, test = split_by_game(data)
    net = NeuralNet(**FAST).fit(train.payoffs, train.n_a, train.n)
    nll_net = metrics(net.predict(test.payoffs), test)["nll"]
    nll_true = metrics(CH_TRUTH.predict(test.payoffs), test)["nll"]
    assert nll_net < np.log(2) - 0.05          # far better than coin flipping
    assert nll_net - nll_true < 0.01           # and close to the true model


def test_network_finds_what_the_theories_miss():
    """Data where 30% of players are prosocial: CH can't express that, the network can."""
    truth = ProsocialMix(CH_TRUTH, share=0.3)
    data = simulate(truth, n_games=1208, seed=5)
    train, test = split_by_game(data)
    net = NeuralNet(**FAST).fit(train.payoffs, train.n_a, train.n)
    ch = CognitiveHierarchy().fit(train.payoffs, train.n_a, train.n)
    nll_net = metrics(net.predict(test.payoffs), test)["nll"]
    nll_ch = metrics(ch.predict(test.payoffs), test)["nll"]
    assert nll_net < nll_ch - 0.005, (nll_net, nll_ch)


def test_all_models_includes_the_network_only_when_asked():
    assert all_models(neural=True)[-1].name == "Neural network"
    assert "Neural network" not in [m.name for m in all_models(neural=False)]
