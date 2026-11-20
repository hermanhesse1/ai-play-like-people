"""One evaluation for every artificial player.

* Cross-validation is grouped by game: both perspectives of a game always sit
  in the same fold, so no model is scored on a game it has seen from the other
  side.
* Main score: negative log likelihood per human decision (lower is better).
* Also MSE of predicted shares and completeness, so numbers line up with the
  2025 paper (which reports MSE and completeness, not likelihood).
* Noise floor: even a perfect model of the true choice probabilities has
  MSE equal to the binomial sampling variance of each observed share. We
  estimate it and report how much of the explainable variance each model gets.
* Uncertainty: bootstrap over games (perspectives resampled together), and
  paired bootstrap for differences between models.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data import GameData
from .games import CATEGORIES
from .models.base import EPS


# ---------------------------------------------------------------- metrics

def per_instance_nll(p, n_a, n):
    p = np.clip(p, EPS, 1 - EPS)
    return -(n_a * np.log(p) + (n - n_a) * np.log(1 - p))


def metrics(p: np.ndarray, data: GameData) -> dict:
    share = data.share
    nll = per_instance_nll(p, data.n_a, data.n).sum() / data.n.sum()
    mse = np.mean((p - share) ** 2)
    var = np.var(share)
    return {"nll": nll, "mse": mse, "mae": np.mean(np.abs(p - share)),
            "r2": 1 - mse / var if var > 0 else np.nan}


def noise_floor(data: GameData) -> dict:
    """Irreducible error from finite samples.

    The unbiased estimate of each share's sampling variance is
    s (1 - s) / (n - 1). Its mean is the MSE a perfect model would still get.
    """
    s, n = data.share, data.n
    floor = np.mean(s * (1 - s) / np.maximum(n - 1, 1))
    var = np.var(s)
    return {"mse_floor": floor, "r2_ceiling": 1 - floor / var}


# ---------------------------------------------------------------- folds

def game_folds(data: GameData, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index for every instance, assigned per underlying game."""
    rng = np.random.default_rng(seed)
    games = np.unique(data.game_id)
    rng.shuffle(games)
    fold_of_game = {g: i % k for i, g in enumerate(games)}
    return np.array([fold_of_game[g] for g in data.game_id])


def cross_validate(model, data: GameData, k: int = 5, seed: int = 0):
    """Out-of-fold predictions and the parameters fitted in each fold."""
    folds = game_folds(data, k, seed)
    oof = np.full(len(data), np.nan)
    fitted = []
    for f in range(k):
        train, test = folds != f, folds == f
        m = copy.deepcopy(model)
        m.fit(data.payoffs[train], data.n_a[train], data.n[train])
        oof[test] = m.predict(data.payoffs[test])
        fitted.append({"fold": f, **m.params()})
    return oof, pd.DataFrame(fitted)


# ---------------------------------------------------------------- bootstrap

def _game_index(data: GameData):
    games, inv = np.unique(data.game_id, return_inverse=True)
    return len(games), inv


def bootstrap(preds: dict[str, np.ndarray], data: GameData, B: int = 1000, seed: int = 0):
    """95% intervals for NLL and MSE of each model, and paired differences vs the best.

    Resamples underlying games with replacement; both perspectives go together.
    Works from per-game sums, so it is fast.
    """
    rng = np.random.default_rng(seed)
    G, inv = _game_index(data)
    n_per_game = np.bincount(inv, weights=data.n, minlength=G)
    inst_per_game = np.bincount(inv, minlength=G)
    sums = {}
    for name, p in preds.items():
        nll_g = np.bincount(inv, weights=per_instance_nll(p, data.n_a, data.n), minlength=G)
        se_g = np.bincount(inv, weights=(p - data.share) ** 2, minlength=G)
        sums[name] = (nll_g, se_g)

    draws = rng.integers(0, G, size=(B, G))
    counts = np.stack([np.bincount(d, minlength=G) for d in draws])      # (B, G)
    tot_n = counts @ n_per_game
    tot_inst = counts @ inst_per_game
    boot = {name: (counts @ nll_g / tot_n, counts @ se_g / tot_inst)
            for name, (nll_g, se_g) in sums.items()}

    point = {name: metrics(p, data) for name, p in preds.items()}
    best = min(point, key=lambda m: point[m]["nll"])
    rows = []
    for name, (b_nll, b_mse) in boot.items():
        diff = b_nll - boot[best][0]
        rows.append({
            "model": name,
            "nll": point[name]["nll"],
            "nll_lo": np.percentile(b_nll, 2.5), "nll_hi": np.percentile(b_nll, 97.5),
            "mse": point[name]["mse"],
            "mse_lo": np.percentile(b_mse, 2.5), "mse_hi": np.percentile(b_mse, 97.5),
            "mae": point[name]["mae"], "r2": point[name]["r2"],
            "nll_minus_best": point[name]["nll"] - point[best]["nll"],
            "diff_lo": np.percentile(diff, 2.5), "diff_hi": np.percentile(diff, 97.5),
        })
    return pd.DataFrame(rows), best


def completeness(table: pd.DataFrame, reference: str, random: str = "Random") -> pd.Series:
    """Share of the gap from random to the reference model that each model closes (MSE).

    The 2025 paper sets random = 0% and its neural network = 100%. Until our
    network exists, pass the best classic model or use completeness_to_floor.
    """
    mse = table.set_index("model")["mse"]
    return 100 * (mse[random] - mse) / (mse[random] - mse[reference])


def completeness_to_floor(table: pd.DataFrame, floor: float, random: str = "Random") -> pd.Series:
    """Same idea, but 100% means reaching the noise floor (a perfect model)."""
    mse = table.set_index("model")["mse"]
    return 100 * (mse[random] - mse) / (mse[random] - floor)


def by_category(preds: dict[str, np.ndarray], data: GameData) -> pd.DataFrame:
    """NLL and MSE for each model within each strategic structure."""
    rows = []
    for cat in CATEGORIES:
        mask = data.category == cat
        if not mask.any():
            continue
        sub = data.subset(mask)
        for name, p in preds.items():
            m = metrics(p[mask], sub)
            rows.append({"category": cat, "instances": int(mask.sum()), "model": name,
                         "nll": m["nll"], "mse": m["mse"]})
    return pd.DataFrame(rows)
