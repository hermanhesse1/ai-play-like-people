"""
start_here.py: Do AI agents play like people? Stage 1 in one file.

WHAT THIS SCRIPT DOES
---------------------
People playing a one-shot 2x2 game choose between two actions, A and B. For
each game we know what share of people chose A. This script fits four classic
theories of strategic behaviour to those shares and measures how well each one
predicts games it has never seen.

    Random               everyone flips a coin (the floor every model must beat)
    Nash equilibrium     fully rational play, softened by a fitted "tremble"
    QRE                  noisy best responses; each player expects the other's noise
    Level-k              people think 0, 1, 2 or 3 steps ahead about the opponent
    Cognitive hierarchy  each level replies to a Poisson mix of the lower levels

Every model is a function with one job: take a game's 8 payoffs and return
P(A), the probability that the deciding player picks A. Because they all share
that shape, one piece of scoring code can compare them fairly, and the neural
network, Q-learning agents and language models can plug in later the same way.

HOW IT WORKS, STEP BY STEP
--------------------------
1. Games.  Each game is 8 numbers seen from the decider's side:
       my payoff      for (A,A), (A,B), (B,A), (B,B)
       their payoff   for (A,A), (A,B), (B,A), (B,B)
   where the first letter is my action and the second is the opponent's.
   Each real game was played from both seats, so it appears twice.
2. Fitting.  Each model's parameters are chosen by maximum likelihood: the
   values under which the choices people actually made are most probable.
3. Cross-validation.  The games are split into 5 groups. Each model is fitted
   on 4 groups and predicts the 5th, five times over. Both seats of a game stay
   in the same group, so no model is tested on a game it has already seen.
4. Scores.
       NLL  average negative log likelihood per human decision; lower is better.
            Random play scores ln 2 = 0.693.
       MSE  mean squared error of the predicted share of people choosing A.
       Explained  how much of the gap between random play and a perfect model
            each model closes. A perfect model still has some error, because
            ~39 people per game is a small sample (the "noise floor").
   95% intervals come from resampling games 1,000 times.
5. Game types.  Scores are also split by the game's structure:
       own_dominant  I have a dominant action, no need to think about the opponent
       opp_dominant  only the opponent has one: one step of reasoning solves it
       no_dominant   neither does (coordination-type games)

HOW TO RUN
----------
    pip install numpy scipy
    python start_here.py                    # synthetic data, about 15 seconds
    python start_here.py --data games.csv   # real data in the format below

Without --data the script invents 1,208 games the way the 2025 paper did, and
generates "human" choices from a known cognitive hierarchy model plus noise.
That checks the machinery: the right models should win and the true
parameters (tau = 1.5, lambda = 0.15) should come back. Synthetic numbers
are not findings.

DATA FORMAT FOR --data
----------------------
A CSV with one row per game seen from one seat, and these columns:
    game_id                    same id for both seats of a game
    u_aa,u_ab,u_ba,u_bb        decider's payoffs
    v_aa,v_ab,v_ba,v_bb        opponent's payoffs, indexed the same way
    n_a                        how many deciders chose A
    n                          how many decided

Data source: Zhu, Peterson, Enke and Griffiths (2025), Nature Human Behaviour,
https://osf.io/xrvaw
"""

import argparse
import csv

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit          # the logistic function, 1 / (1 + e^-x)
from scipy.stats import poisson

EPS = 1e-6                               # keeps probabilities away from exactly 0 and 1
TYPES = ["own_dominant", "opp_dominant", "no_dominant"]


# =============================================================================
# 1. Games
# =============================================================================

def split(P):
    """(N, 8) payoff rows -> U, V of shape (N, 2, 2).

    U[n, i, j] is my payoff and V[n, i, j] the opponent's, when I play i and
    they play j (0 = A, 1 = B).
    """
    return P[:, :4].reshape(-1, 2, 2), P[:, 4:].reshape(-1, 2, 2)


def flip(P):
    """The same games seen from the other seat."""
    U, V = split(P)
    their_U = V.transpose(0, 2, 1).reshape(-1, 4)
    their_V = U.transpose(0, 2, 1).reshape(-1, 4)
    return np.concatenate([their_U, their_V], axis=1)


def gain_coefficients(P):
    """How much better A is than B, for each player, as a straight line.

    If the opponent plays A with probability q, my gain from A over B is a*q + b.
    If I play A with probability p, their gain from A over B is c*p + d.
    Every model below is built from these four numbers.
    """
    U, V = split(P)
    a = (U[:, 0, 0] - U[:, 1, 0]) - (U[:, 0, 1] - U[:, 1, 1])
    b = U[:, 0, 1] - U[:, 1, 1]
    c = (V[:, 0, 0] - V[:, 0, 1]) - (V[:, 1, 0] - V[:, 1, 1])
    d = V[:, 1, 0] - V[:, 1, 1]
    return a, b, c, d


def game_type(P):
    """Label each game own_dominant, opp_dominant or no_dominant."""
    U, V = split(P)
    i_dominate = ((U[:, 0, 0] > U[:, 1, 0]) & (U[:, 0, 1] > U[:, 1, 1])) | \
                 ((U[:, 0, 0] < U[:, 1, 0]) & (U[:, 0, 1] < U[:, 1, 1]))
    they_dominate = ((V[:, 0, 0] > V[:, 0, 1]) & (V[:, 1, 0] > V[:, 1, 1])) | \
                    ((V[:, 0, 0] < V[:, 0, 1]) & (V[:, 1, 0] < V[:, 1, 1]))
    return np.where(i_dominate, TYPES[0], np.where(they_dominate, TYPES[1], TYPES[2]))


def has_pure_equilibrium(P):
    U, V = split(P)
    i_best = U >= U[:, ::-1, :]            # my action i is a best reply to their j
    they_best = V >= V[:, :, ::-1]         # their action j is a best reply to my i
    return (i_best & they_best).any(axis=(1, 2))


def generate_games(n_games, rng):
    """Random games made the way the 2025 paper made them.

    Each payoff is drawn from {1..u}, with u itself drawn from {1..50}, and
    games with no pure-strategy equilibrium are thrown away.
    """
    games = []
    while len(games) < n_games:
        u = rng.integers(1, 51, size=(4 * n_games, 8))
        P = rng.integers(1, u + 1).astype(float)
        games.extend(P[has_pure_equilibrium(P)])
    return np.array(games[:n_games])


# =============================================================================
# 2. Data
# =============================================================================

class Data:
    def __init__(self, payoffs, n_a, n, game_id):
        self.payoffs = np.asarray(payoffs, dtype=float)
        self.n_a = np.asarray(n_a, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.game_id = np.asarray(game_id)
        self.share = self.n_a / self.n
        self.type = game_type(self.payoffs)

    def __len__(self):
        return len(self.n)

    def subset(self, mask):
        return Data(self.payoffs[mask], self.n_a[mask], self.n[mask], self.game_id[mask])


def load_csv(path):
    cols = ["u_aa", "u_ab", "u_ba", "u_bb", "v_aa", "v_ab", "v_ba", "v_bb"]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return Data([[float(r[c]) for c in cols] for r in rows],
                [float(r["n_a"]) for r in rows],
                [float(r["n"]) for r in rows],
                [r["game_id"] for r in rows])


def simulate(predict, n_games=1208, people_per_game=39, noise=0.6, seed=0):
    """Fake data: choices drawn from a known model, plus noise it can't explain."""
    rng = np.random.default_rng(seed)
    games = generate_games(n_games, rng)
    payoffs = np.concatenate([games, flip(games)])            # both seats
    game_id = np.concatenate([np.arange(n_games)] * 2)
    p = np.clip(predict(payoffs), EPS, 1 - EPS)
    p = expit(np.log(p / (1 - p)) + rng.normal(0, noise, len(p)))
    n = np.maximum(rng.poisson(people_per_game, len(p)), 5)
    return Data(payoffs, rng.binomial(n, p), n, game_id)


# =============================================================================
# 3. The models: each turns payoffs into P(A)
# =============================================================================

def nash_p(P):
    """Average of the decider's P(A) over every Nash equilibrium of each game.

    Pure equilibria are cells where both players are best-replying. The mixed
    equilibrium, when there is one, makes each player indifferent: the
    opponent plays A with q* = -b/a, and I play A with p* = -d/c.
    """
    U, V = split(P)
    pure = (U >= U[:, ::-1, :]) & (V >= V[:, :, ::-1])
    a, b, c, d = gain_coefficients(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_star, p_star = -b / a, -d / c
    mixed = (q_star > 0) & (q_star < 1) & (p_star > 0) & (p_star < 1)
    total_pa = pure[:, 0, :].sum(axis=1) + np.where(mixed, p_star, 0)
    count = pure.sum(axis=(1, 2)) + mixed
    return total_pa / np.maximum(count, 1)


def qre_equation(x, lam, a, b, c, d):
    """The QRE condition h(x) = 0 for x = logit(p), with its two slopes.

    h      = x - lam * (a*q + b), where q = expit(lam * (c*p + d)) and p = expit(x)
    h_x    = dh/dx    (positive at a stable solution)
    h_lam  = dh/dlam  (used to predict where the solution moves as lam grows)
    """
    p = expit(x)
    u = c * p + d
    q = expit(lam * u)
    h = x - lam * (a * q + b)
    h_x = 1 - lam * a * q * (1 - q) * lam * c * p * (1 - p)
    h_lam = -(a * q + b) - lam * a * q * (1 - q) * u
    return h, h_x, h_lam


def stable_root_near(x0, lam, a, b, c, d):
    """For one game: the stable solution closest to x0, found by scanning and bisection."""
    bound = lam * (abs(a) + abs(b)) + 1          # every solution lies in [-bound, bound]
    xs = np.linspace(-bound, bound, 20001)
    hs = qre_equation(xs, lam, a, b, c, d)[0]
    upward = np.where((hs[:-1] < 0) & (hs[1:] >= 0))[0]     # h rising through 0 = stable
    f = lambda x: float(qre_equation(np.array(x), lam, a, b, c, d)[0])
    roots = [brentq(f, xs[i], xs[i + 1], xtol=1e-13) for i in upward]
    return min(roots, key=lambda r: abs(r - x0))


def qre_p(P, lam, dlam=0.002):
    """Logit quantal response equilibrium.

    Each player picks A with probability expit(lam * gain from A), and each
    expects the other to do the same, so the answer is a fixed point:
        p = expit(lam * (a*q + b)),  q = expit(lam * (c*p + d)).
    Substituting q leaves one equation in x = logit(p) (see qre_equation).

    Coordination games can have three solutions once lam is large enough. The
    standard choice is the "principal branch": the solution you reach by
    starting at lam = 0 (everyone 50/50) and raising lam continuously. We do
    exactly that, carefully:
      1. take small steps in lam (at most dlam),
      2. predict each new solution from the slope dx/dlam = -h_lam / h_x,
      3. polish the prediction with Newton's method,
      4. accept it only if it is a stable solution (h_x > 0); otherwise search
         for the stable solution nearest the prediction. This also handles the
         rare "fold", where the branch being followed simply ends and only one
         solution is left.
    Larger steps without step 4 can jump to the wrong solution in a few games.
    """
    a, b, c, d = gain_coefficients(P)
    x = np.zeros(len(P))
    grid = np.linspace(0, lam, int(np.ceil(lam / dlam)) + 1)
    for lam0, lam1 in zip(grid[:-1], grid[1:]):
        _, h_x, h_lam = qre_equation(x, lam0, a, b, c, d)
        slope = -h_lam / np.where(np.abs(h_x) < 1e-3, 1e-3, h_x)
        x_pred = x + np.clip(slope * (lam1 - lam0), -1, 1)
        x = x_pred.copy()
        for _ in range(8):
            h, h_x, _ = qre_equation(x, lam1, a, b, c, d)
            x = x - np.clip(h / np.where(np.abs(h_x) < 1e-9, 1e-9, h_x), -1, 1)
        h, h_x, _ = qre_equation(x, lam1, a, b, c, d)
        for i in np.where(~((np.abs(h) < 1e-9) & (h_x > 0)))[0]:
            x[i] = stable_root_near(x_pred[i], lam1, a[i], b[i], c[i], d[i])
    return expit(x)


def level_k_p(P, lam, weights):
    """Level 0 plays 50/50; level k noisily best-replies to level k-1.

    weights[k] is the share of people at level k. Both players' levels are
    built side by side, because my level 2 replies to their level 1, which
    replies to my level 0.
    """
    a, b, c, d = gain_coefficients(P)
    mine, theirs = [np.full(len(P), 0.5)], [np.full(len(P), 0.5)]
    for k in range(1, len(weights)):
        mine.append(expit(lam * (a * theirs[k - 1] + b)))
        theirs.append(expit(lam * (c * mine[k - 1] + d)))
    return np.asarray(weights) @ np.array(mine)


def cognitive_hierarchy_p(P, lam, tau, max_level=7):
    """Like level-k, but level k replies to a Poisson(tau) mix of levels 0..k-1."""
    a, b, c, d = gain_coefficients(P)
    f = poisson.pmf(np.arange(max_level + 1), tau)
    mine, theirs = [np.full(len(P), 0.5)], [np.full(len(P), 0.5)]
    for k in range(1, max_level + 1):
        w = f[:k] / f[:k].sum()                      # beliefs about lower levels
        q = w @ np.array(theirs)
        p = w @ np.array(mine)
        mine.append(expit(lam * (a * q + b)))
        theirs.append(expit(lam * (c * p + d)))
    return (f / f.sum()) @ np.array(mine)


class Model:
    """A named model with free parameters theta, fitted by maximum likelihood.

    theta is unconstrained (any real numbers), and `predict` maps it onto the
    allowed range, e.g. lam = exp(theta[0]) > 0, so the optimiser can't step
    outside it.
    """

    def __init__(self, name, predict, starts, readable, bounds=None):
        self.name = name
        self._predict = predict          # (payoffs, theta) -> P(A)
        self.starts = starts             # starting points for the optimiser
        self.readable = readable         # theta -> dict of named parameters
        self.bounds = bounds             # optional limits on theta
        self.theta = np.array(starts[0]) if starts else np.array([])

    def predict(self, P):
        return self._predict(P, self.theta)

    def fit(self, data):
        if not self.starts:
            return self
        loss = lambda th: nll(self._predict(data.payoffs, th), data)
        best = min((minimize(loss, s, method="L-BFGS-B", bounds=self.bounds)
                    for s in self.starts), key=lambda r: r.fun)
        self.theta = best.x
        return self

    def params(self):
        return self.readable(self.theta)


def softmax_weights(z):
    z = np.concatenate([[0.0], z])        # level 0 fixed at 0 so weights are identified
    e = np.exp(z - z.max())
    return e / e.sum()


def make_models():
    return [
        Model("Random", lambda P, th: np.full(len(P), 0.5), [], lambda th: {}),
        Model("Nash",
              lambda P, th: (1 - expit(th[0])) * nash_p(P) + expit(th[0]) / 2,
              [[-1.0]], lambda th: {"eps": expit(th[0])}),
        Model("QRE",
              lambda P, th: qre_p(P, np.exp(th[0])),
              [[np.log(0.05)], [np.log(0.3)]], lambda th: {"lambda": np.exp(th[0])},
              bounds=[(np.log(1e-4), np.log(5.0))]),    # keeps each QRE solve quick
        Model("Level-k",
              lambda P, th: level_k_p(P, np.exp(th[0]), softmax_weights(th[1:])),
              [[np.log(0.2), 0, 0, 0], [np.log(0.05), 1, 1, 1]],
              lambda th: {"lambda": np.exp(th[0]),
                          **{f"level {k}": w for k, w in enumerate(softmax_weights(th[1:]))}}),
        Model("Cognitive hierarchy",
              lambda P, th: cognitive_hierarchy_p(P, np.exp(th[0]), np.exp(th[1])),
              [[np.log(0.2), np.log(1.5)], [np.log(0.05), np.log(4.0)]],
              lambda th: {"lambda": np.exp(th[0]), "tau": np.exp(th[1])}),
    ]


# =============================================================================
# 4. Scoring
# =============================================================================

def nll_per_game(p, data):
    p = np.clip(p, EPS, 1 - EPS)
    return -(data.n_a * np.log(p) + (data.n - data.n_a) * np.log(1 - p))


def nll(p, data):
    """Average negative log likelihood per human decision."""
    return nll_per_game(p, data).sum() / data.n.sum()


def mse(p, data):
    return np.mean((p - data.share) ** 2)


def noise_floor(data):
    """MSE a perfect model would still get, from sampling noise alone."""
    s = data.share
    return np.mean(s * (1 - s) / np.maximum(data.n - 1, 1))


def cross_validate(model, data, k=5, seed=0):
    """Out-of-sample predictions for every game, from 5 fits on the other 4/5."""
    rng = np.random.default_rng(seed)
    games = np.unique(data.game_id)
    rng.shuffle(games)
    fold_of = {g: i % k for i, g in enumerate(games)}
    fold = np.array([fold_of[g] for g in data.game_id])
    out = np.empty(len(data))
    for f in range(k):
        model.fit(data.subset(fold != f))
        out[fold == f] = model.predict(data.payoffs[fold == f])
    return out


def bootstrap_interval(p, data, draws=1000, seed=0):
    """95% interval for NLL, resampling whole games (both seats together)."""
    rng = np.random.default_rng(seed)
    games, idx = np.unique(data.game_id, return_inverse=True)
    loss_g = np.bincount(idx, weights=nll_per_game(p, data))
    n_g = np.bincount(idx, weights=data.n)
    counts = np.array([np.bincount(rng.integers(0, len(games), len(games)),
                                   minlength=len(games)) for _ in range(draws)])
    samples = (counts @ loss_g) / (counts @ n_g)
    return np.percentile(samples, [2.5, 97.5])


# =============================================================================
# 5. Run everything and print the results
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Fit and score the classic models.")
    ap.add_argument("--data", help="CSV in the format described at the top")
    args = ap.parse_args()

    if args.data:
        data = load_csv(args.data)
        print(f"Real data from {args.data}")
    else:
        truth = lambda P: cognitive_hierarchy_p(P, lam=0.15, tau=1.5)
        data = simulate(truth)
        print("SYNTHETIC data from cognitive hierarchy (tau = 1.5, lambda = 0.15). Not findings.")
    print(f"{len(data)} game instances, {len(np.unique(data.game_id))} games, "
          f"{int(data.n.sum())} decisions\n")

    models = make_models()
    predictions = {}
    for m in models:
        print(f"  cross-validating {m.name} ...")
        predictions[m.name] = cross_validate(m, data)

    floor = noise_floor(data)
    random_mse = mse(predictions["Random"], data)
    rows = []
    for name, p in predictions.items():
        lo, hi = bootstrap_interval(p, data)
        explained = 100 * (random_mse - mse(p, data)) / (random_mse - floor)
        rows.append((nll(p, data), name, lo, hi, mse(p, data), explained))
    rows.sort()

    print(f"\n{'Model':<22}{'NLL':>8}   {'95% interval':<18}{'MSE':>8}{'Explained':>11}")
    for score, name, lo, hi, err, explained in rows:
        print(f"{name:<22}{score:8.4f}   [{lo:.4f}, {hi:.4f}]  {err:8.4f}{explained:10.0f}%")
    print(f"\nNoise floor: a perfect model would still have MSE of about {floor:.4f}.")

    print(f"\nNLL by game type")
    present = [t for t in TYPES if (data.type == t).any()]
    print(f"{'Model':<22}" + "".join(f"{t:>15}" for t in present))
    print(f"{'(instances)':<22}" + "".join(f"{(data.type == t).sum():>15}" for t in present))
    for _, name, *_ in rows:
        p = predictions[name]
        cells = [nll(p[data.type == t], data.subset(data.type == t)) for t in present]
        print(f"{name:<22}" + "".join(f"{v:15.4f}" for v in cells))

    print("\nParameters fitted on all the data")
    for m in models:
        m.fit(data)
        pretty = ", ".join(f"{k} = {v:.3f}" for k, v in m.params().items()) or "none"
        print(f"  {m.name:<22}{pretty}")


if __name__ == "__main__":
    main()
