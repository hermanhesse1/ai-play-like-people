# Do AI agents play like people?

Which artificial players reproduce human choices in one-shot 2×2 games, and in which kinds of games does each one fail?

This repository compares classic game theory models, a neural network, reinforcement learning agents and language models against **93,460 human decisions in 2,416 game instances** (1,208 games, each seen from both players' sides) from Zhu, Peterson, Enke and Griffiths (2025).

## Status

| Stage | Players | State |
|---|---|---|
| 1 | Random, Nash, QRE, level-k, cognitive hierarchy | Built and tested on synthetic data; waiting for the human data |
| 2 | Neural network (PyTorch) | Next |
| 3 | Q-learning agents | Planned |
| 4 | Language models | Planned |

## Quick start

```bash
pip install -r requirements.txt
python start_here.py   # the whole of stage 1 in one commented file (~10 s)
make test              # 22 tests: textbook games, QRE branches, parameter recovery
make synthetic         # full pipeline + figures on synthetic data from a known model (~1 min)
make all               # the real run: tests, models, figures (needs data/games2p2k.csv, see data/README.md)
```

`start_here.py` is the easiest way in: one file, only NumPy and SciPy, with a description at the top of what each step does. The package under `src/` is the same models split into modules, with bootstrap differences, CSV outputs and tests.

Results go to `results/<data file name>/`: `results.md` (the table), `summary.csv`, `by_category.csv`, `params.csv`, `oof_predictions.csv` and the parameters fitted in each fold.

`scripts/make_figures.py` turns `oof_predictions.csv` into the paper's three figures, as PNG and PDF:

1. `fig1_scores`: how much better than random play each model predicts people, with 95% intervals.
2. `fig2_predicted_vs_observed`: the best model's predictions against human shares, one panel per game type.
3. `fig3_by_game_type`: figure 1 within each game type; bars left of zero mean worse than random.

Scores are plotted as ln 2 − NLL, the improvement over coin flipping. Raw NLL values all sit between about 0.55 and 0.69, so bars of them would look alike; the improvement has a true zero.

## How every player is scored

All players are scored the same way, so numbers are directly comparable.

* **Cross-validation grouped by game.** 5 folds. Both perspectives of a game always sit in the same fold, so no model is tested on a game it was fitted on from the other side.
* **Main score: negative log likelihood per decision** (lower is better). Random play scores ln 2 ≈ 0.693.
* **MSE, R² and completeness** of predicted shares, because that is how the 2025 paper reports its models, so the two can be compared.
* **Noise floor.** With about 39 people per instance, even a perfect model has MSE equal to the sampling variance of the observed shares. The table reports how much of the gap between random play and that floor each model closes.
* **Uncertainty.** 95% intervals from 1,000 bootstrap draws over games, and paired intervals for each model's difference from the best one.
* **Breakdown by strategic structure**, from the decider's point of view:
  * `own_dominant`: I have a dominant action, no reasoning about the opponent needed
  * `opp_dominant`: only the opponent does, so one step of reasoning about them solves the game
  * `no_dominant`: neither does; in this dataset these are games with two pure equilibria

  The dataset has no games with only mixed equilibria: the authors dropped every game without a pure equilibrium.

## The classic models

| Model | Parameters | Notes |
|---|---|---|
| Random | 0 | Floor |
| Nash | 1 (tremble ε) | Averages all equilibria, pure and mixed. A pure prediction of 0 or 1 has infinite log likelihood, so it is mixed with uniform play by a fitted ε. The untrembled MSE is also reported, as in the paper. |
| QRE | 1 (λ) | Logit quantal response equilibrium on the principal branch, followed from λ = 0 in small steps with a tangent predictor and Newton corrector. Only stable solutions are accepted, which keeps the path from jumping to another solution and carries it across the rare fold. |
| Level-k | 4 (λ, 3 weights) | Levels 0 to 3, level 0 uniform, logit responses. |
| Cognitive hierarchy | 2 (τ, λ) | Poisson mix of lower levels, truncated at level 7, logit responses. |

All are fitted by maximum likelihood of the observed counts.

## Layout

```
src/humanplay/
  games.py          game representation, perspective flip, dominance, categories
  data.py           loading, synthetic games generated as in the paper
  evaluate.py       grouped CV, metrics, noise floor, bootstrap, breakdowns
  models/           nash.py, qre.py, levelk.py (level-k and CH), base.py
scripts/
  prepare_data.py   decision-level OSF file -> canonical CSV
  run_classic.py    fit and score the classic models
tests/              pytest suite
```

## Data

Zhu, J.-Q., Peterson, J. C., Enke, B. and Griffiths, T. L. (2025). Capturing the complexity of human strategic decision-making with machine learning. *Nature Human Behaviour*. https://doi.org/10.1038/s41562-025-02230-5. Data and code: https://osf.io/xrvaw. See `data/README.md`.

## Licence

Code: MIT. The data keep the licence set by their authors on OSF.
