"""The three figures for the paper, built from one results file.

    python scripts/make_figures.py results/synthetic      # after make synthetic
    python scripts/make_figures.py results/synthetic_prosocial
    python scripts/make_figures.py results/games2p2k      # after make classic

Reads <folder>/oof_predictions.csv (written by run_models.py: every game
instance with each model's out-of-fold prediction) and writes, as PNG and PDF:

  fig1_scores            How much better than random play each model predicts
                         people, with 95% intervals over games.
  fig2_predicted_vs_observed
                         The best model's prediction against the share of
                         people who actually chose A, one dot per game instance,
                         split by game type. Later: the best language model.
  fig3_by_game_type      Figure 1 repeated within each game type, to show
                         where each model fails.

Why "improvement over random" and not raw NLL: every model's NLL sits between
about 0.55 and 0.69, so bars of raw NLL would all look the same length.
Random play has NLL = ln 2, so ln 2 - NLL is zero for coin flipping and grows as
a model explains more. It has a true zero, so bars are honest, and a model that
does worse than random gets a bar pointing left.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from humanplay.data import GameData  # noqa: E402
from humanplay.evaluate import bootstrap, metrics  # noqa: E402
from humanplay.games import PAYOFF_COLUMNS  # noqa: E402

LN2 = np.log(2)
TYPE_NAMES = {"own_dominant": "I have a dominant action",
              "opp_dominant": "Only the opponent has one",
              "no_dominant": "Neither has one"}

# Palette (light mode): one series hue, ink for text, recessive chrome.
BLUE, INK, INK2, MUTED = "#2a78d6", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
})


def load(folder: Path):
    df = pd.read_csv(folder / "oof_predictions.csv")
    data = GameData(df[PAYOFF_COLUMNS].to_numpy(), df.n_a, df.n, df.game_id,
                    source=str(folder))
    preds = {c[5:].replace("_", " "): df[c].to_numpy()
             for c in df.columns if c.startswith("pred_")}
    return data, preds


def improvement_table(preds, data, seed=0):
    """ln 2 - NLL for each model, with a 95% game-level bootstrap interval."""
    table, _ = bootstrap(preds, data, B=1000, seed=seed)
    table["gain"] = LN2 - table.nll
    table["gain_lo"] = LN2 - table.nll_hi
    table["gain_hi"] = LN2 - table.nll_lo
    return table.set_index("model")


def style(ax, xlabel):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=BASE, linewidth=1)
    ax.set_xlabel(xlabel)


def bars(ax, table, order, label_values=True):
    y = np.arange(len(order))[::-1]
    g = table.loc[order]
    ax.barh(y, g.gain, height=0.5, color=BLUE)
    ax.errorbar(g.gain, y, xerr=[g.gain - g.gain_lo, g.gain_hi - g.gain],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.set_yticks(y, order)
    if label_values:
        span = max(g.gain_hi.max(), 0) - min(g.gain_lo.min(), 0)
        for yi, (lo, hi, v) in zip(y, zip(g.gain_lo, g.gain_hi, g.gain)):
            x = (hi if v >= 0 else lo) + (0.02 if v >= 0 else -0.02) * span
            ax.text(x, yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                    color=INK2, fontsize=9)


def caveat(fig, synthetic):
    if synthetic:
        fig.text(0.99, -0.02, "Synthetic data: checks the pipeline, not a finding",
                 ha="right", va="top", color=MUTED, fontsize=8, style="italic")


def save(fig, folder, name):
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "synthetic")
    synthetic = folder.name.startswith("synthetic")
    data, preds = load(folder)
    overall = improvement_table(preds, data)
    order = [m for m in overall.sort_values("gain", ascending=False).index if m != "Random"]
    best = order[0]
    unit = "Improvement over random play (nats per decision)"

    # ---- Figure 1: overall scores ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.4))
    bars(ax, overall, order)
    style(ax, unit)
    lo = min(0, overall.loc[order].gain_lo.min())
    ax.set_xlim(lo * 1.25 if lo < 0 else 0, overall.loc[order].gain_hi.max() * 1.18)
    fig.suptitle("How much better than coin flipping each model predicts people",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_title(f"Cross-validated on {len(data):,} game instances; "
                 "whiskers show 95% intervals over games",
                 fontsize=9, fontweight="normal", color=INK2, pad=10)
    fig.tight_layout()
    caveat(fig, synthetic)
    save(fig, folder, "fig1_scores")

    # ---- Figure 2: best model, predicted vs observed ---------------------------
    types = [t for t in TYPE_NAMES if (data.category == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 3.6),
                             sharex=True, sharey=True)
    for ax, t in zip(np.atleast_1d(axes), types):
        mask = data.category == t
        p, s = preds[best][mask], data.share[mask]
        ax.plot([0, 1], [0, 1], color=BASE, linewidth=1, zorder=1)
        ax.scatter(p, s, s=7, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
        mse = np.mean((p - s) ** 2)
        ax.set_title(TYPE_NAMES[t], fontsize=10)
        ax.text(0.04, 0.96, f"{mask.sum():,} instances\nMSE {mse:.4f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", colors=MUTED)
        ax.set_xlabel(f"{best}: predicted P(A)")
    np.atleast_1d(axes)[0].set_ylabel("People who chose A (share)")
    fig.suptitle(f"{best} against human choices; dots on the diagonal are perfect predictions",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    caveat(fig, synthetic)
    save(fig, folder, "fig2_predicted_vs_observed")

    # ---- Figure 3: scores within each game type -------------------------------
    tables = {t: improvement_table({m: p[data.category == t] for m, p in preds.items()},
                                   data.subset(data.category == t)) for t in types}
    lo = min(0, min(tb.loc[order].gain_lo.min() for tb in tables.values()))
    hi = max(tb.loc[order].gain_hi.max() for tb in tables.values())
    fig, axes = plt.subplots(1, len(types), figsize=(3.3 * len(types), 0.5 * len(order) + 1.6),
                             sharex=True, sharey=True)
    for i, (ax, t) in enumerate(zip(np.atleast_1d(axes), types)):
        bars(ax, tables[t], order, label_values=False)
        style(ax, "Improvement over random" if i == 1 else "")
        ax.set_xlim(lo * 1.15 if lo < 0 else 0, hi * 1.1)
        ax.set_title(f"{TYPE_NAMES[t]}\n({(data.category == t).sum():,} instances)",
                     fontsize=10)
    fig.suptitle("Where each model succeeds and fails, by game type "
                 "(bars left of zero: worse than random)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    caveat(fig, synthetic)
    save(fig, folder, "fig3_by_game_type")

    print(f"wrote fig1_scores, fig2_predicted_vs_observed, fig3_by_game_type to {folder}")
    print(f"best model: {best} (NLL {metrics(preds[best], data)['nll']:.4f})")


if __name__ == "__main__":
    main()
