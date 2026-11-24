"""Fit and score every player: the classic models and the neural network.

    python scripts/run_models.py --data data/games2p2k.csv       # real data
    python scripts/run_models.py                                 # synthetic check
    python scripts/run_models.py --classic-only                  # stage 1 only, no PyTorch

Players: Random, Nash, QRE, level-k, cognitive hierarchy (stage 1) and a
neural network trained on the raw payoffs (stage 2).

Without --data it builds a synthetic dataset whose choices come from a known
cognitive hierarchy model plus unexplained noise, so you can see the pipeline
pick the right model and recover its parameters. With --truth prosocial, 30% of
the synthetic players also care about the other player's payoff, which no
classic model can express: the check that the network finds what theories
miss. Synthetic results are written to results/synthetic*/ and must never be
reported as findings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanplay.data import ProsocialMix, load_csv, simulate                       # noqa: E402
from humanplay.evaluate import (bootstrap, by_category, completeness,  # noqa: E402
                                completeness_to_floor, cross_validate, metrics, noise_floor)
from humanplay.models import CognitiveHierarchy, Nash, all_models  # noqa: E402

# Synthetic truths: "ch" is a classic model the theories can fit; "prosocial"
# adds behaviour they can't express, which only the network can pick up.
CH_TRUTH = CognitiveHierarchy(tau=1.5, lam=0.15)
TRUTHS = {"ch": (CH_TRUTH, 0.6), "prosocial": (ProsocialMix(CH_TRUTH, share=0.3), 0.0)}
NN = "Neural network"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="canonical CSV (see data/README.md); omit for synthetic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", choices=TRUTHS, default="ch",
                    help="which known model generates the synthetic data")
    ap.add_argument("--classic-only", action="store_true",
                    help="skip the neural network (no PyTorch needed)")
    args = ap.parse_args()

    if args.data:
        data, tag = load_csv(args.data), Path(args.data).stem
    else:
        truth, noise = TRUTHS[args.truth]
        data = simulate(truth, extra_noise=noise, seed=args.seed)
        tag = "synthetic" if args.truth == "ch" else f"synthetic_{args.truth}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"{tag}: {len(data)} instances, {len(np.unique(data.game_id))} games, "
          f"{int(data.n.sum())} decisions")
    floor = noise_floor(data)

    preds, params = {}, []
    for model in all_models(neural=not args.classic_only):
        t = time.time()
        oof, fold_params = cross_validate(model, data, k=args.folds, seed=args.seed)
        preds[model.name] = oof
        full = model.fit(data.payoffs, data.n_a, data.n)       # parameters on all data
        params.append({"model": model.name, "n_params": model.n_params, **full.params()})
        fold_params.insert(0, "model", model.name)
        fold_params.to_csv(out / f"fold_params_{model.name.replace(' ', '_')}.csv", index=False)
        print(f"  {model!r:<60} cv nll {metrics(oof, data)['nll']:.4f}  ({time.time() - t:.1f}s)")

    # The paper scores Nash without trembles; keep that number for comparison.
    raw_nash = Nash().predict_raw(data.payoffs)
    raw_nash_mse = metrics(raw_nash, data)["mse"]

    table, best = bootstrap(preds, data, B=args.boot, seed=args.seed)
    table["completeness_vs_best"] = completeness(table, reference=best).values
    table["completeness_vs_floor"] = completeness_to_floor(table, floor["mse_floor"]).values
    if NN in preds:      # the 2025 paper's completeness: random = 0%, network = 100%
        table["completeness_vs_nn"] = completeness(table, reference=NN).values
    table = table.sort_values("nll").reset_index(drop=True)
    cats = by_category(preds, data)

    table.to_csv(out / "summary.csv", index=False)
    cats.to_csv(out / "by_category.csv", index=False)
    pd.DataFrame(params).to_csv(out / "params.csv", index=False)
    oof_df = data.to_frame()
    oof_df["category"] = data.category
    for name, p in preds.items():
        oof_df[f"pred_{name.replace(' ', '_')}"] = p
    oof_df.to_csv(out / "oof_predictions.csv", index=False)

    md = report(tag, data, table, cats, params, floor, raw_nash_mse, best,
                None if args.data else TRUTHS[args.truth])
    (out / "results.md").write_text(md)
    print("\n" + md)


def report(tag, data, table, cats, params, floor, raw_nash_mse, best, truth=None):
    has_nn = "completeness_vs_nn" in table.columns
    lines = [f"# Models: {tag} data", ""]
    if truth is not None:
        model, noise = truth
        extra = f" plus logit noise (sd {noise:g})" if noise else ""
        lines += [f"Synthetic check. Choices were generated by {model!r}{extra}. "
                  "These are not findings.", ""]
    lines += [f"{len(data)} instances from {len(np.unique(data.game_id))} games, "
              f"{int(data.n.sum())} decisions. Grouped 5-fold cross-validation by game; "
              "95% intervals from 1,000 game-level bootstrap draws.", "",
              "| Model | NLL per decision | MSE | R² | vs best: NLL diff | % of explainable |"
              + (" % of network |" if has_nn else ""),
              "|---|---|---|---|---|---|" + ("---|" if has_nn else "")]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.model} | {r.nll:.4f} [{r.nll_lo:.4f}, {r.nll_hi:.4f}] "
            f"| {r.mse:.4f} [{r.mse_lo:.4f}, {r.mse_hi:.4f}] | {r.r2:.3f} "
            f"| {r.nll_minus_best:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}] "
            f"| {r.completeness_vs_floor:.0f}% |"
            + (f" {r.completeness_vs_nn:.0f}% |" if has_nn else ""))
    lines += ["",
              f"Noise floor: a perfect model would still have MSE ≈ {floor['mse_floor']:.4f} "
              f"(R² ceiling {floor['r2_ceiling']:.3f}). '% of explainable' is the share of the "
              "gap from random play to that floor that each model closes."
              + (" '% of network' is the paper's completeness: the share of the network's "
                 "MSE gain over random play that each model achieves." if has_nn else ""),
              f"Nash without trembles (as the 2025 paper scores it): MSE {raw_nash_mse:.4f}.",
              f"Best model by NLL: {best}.", "",
              "## By strategic structure (NLL per decision)", ""]
    pivot = cats.pivot(index="model", columns="category", values="nll")
    counts = cats.drop_duplicates("category").set_index("category")["instances"]
    cols = [c for c in ("own_dominant", "opp_dominant", "no_dominant") if c in pivot.columns]
    lines.append("| Model | " + " | ".join(f"{c} ({counts[c]})" for c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for model in table.model:
        lines.append(f"| {model} | " + " | ".join(f"{pivot.loc[model, c]:.4f}" for c in cols) + " |")
    lines += ["", "## Parameters fitted on all data", ""]
    for p in params:
        vals = ", ".join(f"{k}={v:.4g}" for k, v in p.items() if k not in ("model", "n_params"))
        lines.append(f"- {p['model']} ({p['n_params']} params): {vals or 'none'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
