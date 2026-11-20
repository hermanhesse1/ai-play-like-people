"""Convert decision-level data into the canonical per-instance CSV.

    python scripts/prepare_data.py data/raw/<file>.csv data/games2p2k.csv \
        --payoff-cols <8 column names> --choice-col <name> --a-value <value>

--payoff-cols must list the 8 payoffs as the decider saw them, in this order:
    my payoff for (A, A), (A, B), (B, A), (B, B),
    then the opponent's payoff for (A, A), (A, B), (B, A), (B, B),
where the first letter is the decider's action and the second the opponent's.

Rows are grouped by payoff vector into instances. The two perspectives of a
game get the same game_id by matching each vector with its flipped version,
so this does not rely on the source file's own ids.

Check the printed summary against the paper: 2,416 instances, 1,208 games,
93,460 decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humanplay.games import PAYOFF_COLUMNS, flip_perspective  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--payoff-cols", nargs=8, required=True)
    ap.add_argument("--choice-col", required=True)
    ap.add_argument("--a-value", required=True, help="value of --choice-col meaning action A")
    args = ap.parse_args()

    raw = pd.read_csv(args.source)
    pay = raw[args.payoff_cols].to_numpy(dtype=float)
    chose_a = (raw[args.choice_col].astype(str) == str(args.a_value)).to_numpy()

    df = pd.DataFrame(pay, columns=PAYOFF_COLUMNS)
    df["chose_a"] = chose_a
    inst = (df.groupby(PAYOFF_COLUMNS, sort=False)["chose_a"]
              .agg(n_a="sum", n="count").reset_index())

    vec = inst[PAYOFF_COLUMNS].to_numpy()
    flipped = flip_perspective(vec)
    keys = [min(tuple(a), tuple(b)) for a, b in zip(vec, flipped)]
    key_to_id = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    inst.insert(0, "game_id", [key_to_id[k] for k in keys])
    inst = inst.sort_values(["game_id"]).reset_index(drop=True)

    sizes = inst.groupby("game_id").size()
    print(f"{len(inst)} instances, {inst.game_id.nunique()} games, {int(inst.n.sum())} decisions")
    print(f"games seen from both sides: {(sizes == 2).sum()}, from one side only: {(sizes == 1).sum()}")
    if (sizes > 2).any():
        print("warning: some games have more than 2 instances; check the column order")
    print(f"share choosing A: mean {(inst.n_a / inst.n).mean():.3f}; "
          f"deciders per instance: median {inst.n.median():.0f}")
    inst.to_csv(args.target, index=False)
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
