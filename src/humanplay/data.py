"""Loading the human data, and generating synthetic games for testing.

Canonical format (one row per game instance, i.e. per game and perspective):

    game_id, u_aa, u_ab, u_ba, u_bb, v_aa, v_ab, v_ba, v_bb, n_a, n

game_id identifies the underlying game, so the two perspectives of the same
game share an id and are always kept in the same cross-validation fold.
n_a is the number of deciding players who chose A, n the number who decided.
See data/README.md for how the OSF files map onto this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .games import PAYOFF_COLUMNS, categorise, flip_perspective, pure_equilibria, split


@dataclass
class GameData:
    payoffs: np.ndarray            # (N, 8) float, decider's perspective
    n_a: np.ndarray                # (N,) count choosing A
    n: np.ndarray                  # (N,) count deciding
    game_id: np.ndarray            # (N,) id of the underlying game
    category: np.ndarray = field(init=False)
    source: str = "unknown"

    def __post_init__(self):
        self.payoffs = np.asarray(self.payoffs, dtype=float)
        self.n_a = np.asarray(self.n_a, dtype=float)
        self.n = np.asarray(self.n, dtype=float)
        self.game_id = np.asarray(self.game_id)
        if not (len(self.payoffs) == len(self.n_a) == len(self.n) == len(self.game_id)):
            raise ValueError("payoffs, n_a, n and game_id must have the same length")
        if np.any(self.n_a > self.n) or np.any(self.n <= 0):
            raise ValueError("need 0 <= n_a <= n and n > 0 for every instance")
        self.category = categorise(self.payoffs)

    def __len__(self) -> int:
        return len(self.n)

    @property
    def share(self) -> np.ndarray:
        return self.n_a / self.n

    def subset(self, mask) -> "GameData":
        return GameData(self.payoffs[mask], self.n_a[mask], self.n[mask],
                        self.game_id[mask], source=self.source)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.payoffs, columns=PAYOFF_COLUMNS)
        df.insert(0, "game_id", self.game_id)
        df["n_a"] = self.n_a.astype(int)
        df["n"] = self.n.astype(int)
        return df


def load_csv(path: str) -> GameData:
    """Load a file already in the canonical format."""
    df = pd.read_csv(path)
    missing = [c for c in ["game_id", *PAYOFF_COLUMNS, "n_a", "n"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; see data/README.md")
    return GameData(df[PAYOFF_COLUMNS].to_numpy(), df["n_a"].to_numpy(),
                    df["n"].to_numpy(), df["game_id"].to_numpy(), source=path)


# --------------------------------------------------------------------------
# Synthetic data, generated the way the paper describes its games
# --------------------------------------------------------------------------

def generate_games(n_games: int, rng: np.random.Generator) -> np.ndarray:
    """Row-player-view payoff vectors for n_games random 2x2 games.

    Follows the paper's procedure: each of the 8 payoffs is drawn from U{1..u}
    with u ~ U{1..50}, and games without a pure-strategy equilibrium are
    dropped. (The paper also rebalances by game topology; we skip that.)
    """
    games = []
    while len(games) < n_games:
        batch = 4 * (n_games - len(games))
        u = rng.integers(1, 51, size=(batch, 8))
        P = rng.integers(1, u + 1).astype(float)
        U, V = split(P)
        keep = pure_equilibria(U, V).any(axis=(1, 2))
        games.extend(P[keep])
    return np.array(games[:n_games])


def both_perspectives(row_view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack each game's row view with its column view; returns payoffs and game ids."""
    ids = np.arange(len(row_view))
    return (np.concatenate([row_view, flip_perspective(row_view)]),
            np.concatenate([ids, ids]))


def simulate(model, n_games: int = 1208, mean_n: float = 38.7,
             extra_noise: float = 0.0, seed: int = 0) -> GameData:
    """Synthetic dataset whose choices come from a known model.

    model       : any fitted model with a predict(payoffs) method
    mean_n      : average deciders per instance (93,460 / 2,416 = 38.7)
    extra_noise : sd of a per-instance shock on the logit scale, standing in for
                  behaviour the model can't explain
    """
    rng = np.random.default_rng(seed)
    payoffs, ids = both_perspectives(generate_games(n_games, rng))
    p = np.clip(model.predict(payoffs), 1e-9, 1 - 1e-9)
    if extra_noise > 0:
        logit = np.log(p / (1 - p)) + rng.normal(0, extra_noise, size=len(p))
        p = 1 / (1 + np.exp(-logit))
    n = np.maximum(rng.poisson(mean_n, size=len(p)), 5)
    n_a = rng.binomial(n, p)
    return GameData(payoffs, n_a, n, ids, source=f"synthetic:{model.name}")


__all__ = ["GameData", "load_csv", "generate_games", "both_perspectives", "simulate"]
