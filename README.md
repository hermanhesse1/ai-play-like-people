# Do AI agents play like people?

[![tests](https://github.com/hermanhesse1/ai-play-like-people/actions/workflows/tests.yml/badge.svg)](https://github.com/hermanhesse1/ai-play-like-people/actions/workflows/tests.yml)

Which artificial players reproduce human choices in one-shot 2×2 games, and in which kinds of games does each one fail?

This repository compares classic game theory models and a neural network (built), and in later stages reinforcement learning agents and language models, against **93,460 human decisions in 2,416 game instances** (1,208 games, each seen from both players' sides) from Zhu, Peterson, Enke and Griffiths (2025).

## Status

| Stage | Players | State |
|---|---|---|
| 1 | Random, Nash, QRE, level-k, cognitive hierarchy | Built and tested on synthetic data; waiting for the human data |
| 2 | Neural network (PyTorch) | Built and tested on synthetic data; waiting for the human data |
| 3 | Q-learning agents | Next |
| 4 | Language models | Planned |

## Quick start

```bash
pip install -r requirements.txt
python start_here.py   # the whole of stage 1 in one commented file (~10 s, synthetic data)
make test              # 29 tests: textbook games, QRE branches, parameter recovery, the network
make synthetic         # full pipeline + figures on synthetic data from a known model (~1 min)
make prosocial         # synthetic data the theories can't describe: the network should win
make all               # the real run: tests, models, figures (needs data/games2p2k.csv, see data/README.md)
make classic           # stage 1 models only, no PyTorch needed
```

PyTorch is in `requirements.txt`. The default install is large because it includes GPU support; the network is small, so the CPU-only build is enough and much smaller: run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing the rest.

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

## The neural network (stage 2)

The classic models each build in a theory of how people reason. The network builds in none: it sees the 8 payoffs of a game and outputs the probability of choosing A. If it predicts people better than every theory, the gap is behaviour the theories miss; if it does not, the theories already capture what is predictable. Zhu et al. (2025) use their network the same way, as the 100% mark of their "completeness" score, and the results table now reports that column too.

| Choice | What we did | Why |
|---|---|---|
| Input | The 8 raw payoffs, standardised | No game theory features, so the network starts with no theory |
| Network | 2 hidden layers of 32 units, ReLU, sigmoid output | Small: about 2,400 instances is little data for a network |
| Loss | The same binomial likelihood as every other model | Scores stay directly comparable |
| Overfitting | 20% of the training *games* held out; training stops when their loss stops improving; weight decay | Holding out whole games, not single rows, stops a game's other perspective from leaking into the check |
| Ensemble | 5 networks from different random starts, averaged | Less sensitive to the random seed |

Two synthetic checks show what the network can and cannot do (`tests/test_neural.py`):

1. **Data from cognitive hierarchy** (`make synthetic`). The network gets close to the true model but does not beat it, which is expected: when a theory is exactly right, its 2 parameters use the data better than the network's 1,377 weights.
2. **Data where 30% of players are prosocial** (`make prosocial`): they lean towards the action with the higher total payoff for both players. Every classic model assumes people care only about their own payoff, so none can fit this. The network does, and beats cognitive hierarchy clearly.

Both are synthetic: they check that the method works, and say nothing about real people.

## Layout

```
src/humanplay/
  games.py          game representation, perspective flip, dominance, categories
  data.py           loading, synthetic games generated as in the paper
  evaluate.py       grouped CV, metrics, noise floor, bootstrap, breakdowns
  models/           nash.py, qre.py, levelk.py (level-k and CH), neural.py, base.py
scripts/
  prepare_data.py   decision-level OSF file -> canonical CSV
  run_models.py     fit and score every model (--classic-only for stage 1 alone)
  make_figures.py   the three figures
tests/              pytest suite
```

## Data

Zhu, J.-Q., Peterson, J. C., Enke, B. and Griffiths, T. L. (2025). Capturing the complexity of human strategic decision-making with machine learning. *Nature Human Behaviour*. https://doi.org/10.1038/s41562-025-02230-5. Data and code: https://osf.io/xrvaw. See `data/README.md`.

## Licence

Code: MIT. The data keep the licence set by their authors on OSF.
