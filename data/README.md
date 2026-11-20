# Data

The human data come from Zhu, Peterson, Enke and Griffiths (2025), hosted at https://osf.io/xrvaw.

## Getting it

1. Download the data files from the OSF project into `data/raw/` (ignored by git).
2. Note the licence shown on the OSF project page, and record it here before publishing anything.
3. Convert to the canonical format below with `scripts/prepare_data.py`, writing `data/games2p2k.csv`.
4. Check the printed summary against the paper: 2,416 instances, 1,208 games, 93,460 decisions.

## Canonical format

One row per game instance (a game seen from one player's side):

| Column | Meaning |
|---|---|
| `game_id` | The underlying game; both perspectives share it |
| `u_aa`, `u_ab`, `u_ba`, `u_bb` | Decider's payoff when the decider plays the first letter and the opponent the second |
| `v_aa`, `v_ab`, `v_ba`, `v_bb` | Opponent's payoff, indexed the same way (decider's action first) |
| `n_a` | Number of deciders who chose A |
| `n` | Number of deciders |

The paper adapted every display to the row player's view, so for each participant "A" is the top row as they saw it.

## What the paper says about the games

* Payoffs are integers from 1 to 50. Each is drawn from U{1..u} with u ~ U{1..50}.
* Games without a pure-strategy equilibrium were removed. Dominance games were down-sampled by type in Robinson and Goforth's topology, giving 1,208 games.
* Every participant played 20 different games once each, against a new random opponent, with no feedback until the end. So this is initial play, not learning.
