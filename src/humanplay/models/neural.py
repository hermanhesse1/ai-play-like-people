"""Stage 2: a neural network that learns human play straight from the payoffs.

The classic models each assume a theory of how people reason (equilibrium,
noisy best response, steps of thinking). The network assumes none. It sees the
8 payoffs of a game and outputs P(decider plays A). Whatever it learns beyond
the classic models is structure in human play that those theories miss, which
is how Zhu, Peterson, Enke and Griffiths (2025) use their network: as a
benchmark of how predictable people are (their "completeness" sets it to 100%).

Design choices, kept simple so every line can be explained:

* Input: the 8 raw payoffs, standardised with the training mean and sd. No
  game theory features, so the network starts with no theory at all.
* A small multilayer perceptron (two hidden layers of 32 units, ReLU), with
  a sigmoid output: the 2-action softmax, as in the classic models.
* Loss: the same binomial negative log likelihood every other model is fitted
  by, so its score is directly comparable.
* Overfitting control: about 2,400 game instances is little data for a
  network. We hold out 20% of the *games* (both perspectives together, as in
  the outer cross-validation), train with Adam and weight decay, and keep the
  weights from the epoch with the best held-out loss (early stopping).
* Ensemble: several networks from different random starts are averaged,
  which makes the predictions much less sensitive to the random seed.

Everything is seeded, so the same data and settings give the same predictions.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as err:          # pragma: no cover - only hit without torch
    raise ImportError(
        "The neural network needs PyTorch. Install it with "
        "`pip install torch` (see the README for a smaller CPU-only install)."
    ) from err

from ..games import flip_perspective
from .base import Model


def game_keys(payoffs: np.ndarray) -> np.ndarray:
    """An id per underlying game, recovered from the payoffs alone.

    A game seen from the other side is flip_perspective of this side, so the
    two rows of one game give the same key: the smaller (as bytes) of the row
    and its flip. Used to split games, not instances, for early stopping.
    """
    P = np.ascontiguousarray(payoffs, dtype=float)
    F = np.ascontiguousarray(flip_perspective(P))
    keys = [min(a.tobytes(), b.tobytes()) for a, b in zip(P, F)]
    _, ids = np.unique(np.array(keys, dtype=object), return_inverse=True)
    return ids


def make_network(n_in: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Payoffs in, one logit out (sigmoid of it is P(A))."""
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def count_nll(logit: torch.Tensor, n_a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Binomial NLL per decision from logits; equals base.binomial_nll on sigmoid(logit)."""
    log_p = nn.functional.logsigmoid(logit)        # log P(A)
    log_q = nn.functional.logsigmoid(-logit)       # log P(B)
    return -(n_a * log_p + (n - n_a) * log_q).sum() / n.sum()


class NeuralNet(Model):
    """Ensemble of small MLPs trained on the binomial likelihood with early stopping."""
    name = "Neural network"

    def __init__(self, hidden: tuple[int, ...] = (32, 32), n_ensemble: int = 5,
                 lr: float = 3e-3, weight_decay: float = 1e-4, max_epochs: int = 3000,
                 patience: int = 200, val_frac: float = 0.2, seed: int = 0):
        self.hidden = tuple(hidden)
        self.n_ensemble = n_ensemble
        self.lr, self.weight_decay = lr, weight_decay
        self.max_epochs, self.patience = max_epochs, patience
        self.val_frac, self.seed = val_frac, seed
        self.nets: list[nn.Sequential] = []
        self.best_epochs: list[int] = []
        self.mean = self.sd = None
        sizes = (8, *self.hidden, 1)
        self.n_params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

    # ------------------------------------------------------------ helpers

    def _inputs(self, payoffs) -> torch.Tensor:
        x = (np.asarray(payoffs, dtype=float) - self.mean) / self.sd
        return torch.as_tensor(x, dtype=torch.float32)

    def _split(self, payoffs, rng):
        """Boolean mask of training instances; the rest is the early stopping set."""
        ids = game_keys(payoffs)
        games = np.unique(ids)
        n_val = int(round(self.val_frac * len(games)))
        val_games = rng.choice(games, size=n_val, replace=False)
        return ~np.isin(ids, val_games)

    def _train_one(self, x, n_a, n, train, seed):
        torch.manual_seed(seed)
        net = make_network(x.shape[1], self.hidden)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        tr, va = torch.as_tensor(train), torch.as_tensor(~train)
        best_loss, best_epoch = np.inf, 0
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            opt.zero_grad()
            logit = net(x[tr]).squeeze(-1)
            loss = count_nll(logit, n_a[tr], n[tr])
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                val = count_nll(net(x[va]).squeeze(-1), n_a[va], n[va]).item()
            if val < best_loss - 1e-6:
                best_loss, best_epoch = val, epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            elif epoch - best_epoch >= self.patience:
                break
        net.load_state_dict(best_state)
        net.eval()
        return net, best_epoch

    # ------------------------------------------------------------ interface

    def fit(self, payoffs, n_a, n):
        payoffs = np.asarray(payoffs, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.mean = payoffs.mean(axis=0)
        self.sd = payoffs.std(axis=0) + 1e-8
        x = self._inputs(payoffs)
        n_a_t = torch.as_tensor(np.asarray(n_a, dtype=float), dtype=torch.float32)
        n_t = torch.as_tensor(np.asarray(n, dtype=float), dtype=torch.float32)
        train = self._split(payoffs, rng)
        self.nets, self.best_epochs = [], []
        for i in range(self.n_ensemble):
            net, epoch = self._train_one(x, n_a_t, n_t, train, seed=self.seed * 1000 + i)
            self.nets.append(net)
            self.best_epochs.append(epoch)
        return self

    def predict(self, payoffs):
        if not self.nets:
            raise RuntimeError("fit the network before predicting")
        x = self._inputs(payoffs)
        with torch.no_grad():
            p = torch.stack([torch.sigmoid(net(x).squeeze(-1)) for net in self.nets])
        return p.mean(dim=0).double().numpy()

    def params(self):
        if not self.best_epochs:
            return {}
        return {"weights_per_net": self.n_params, "nets": self.n_ensemble,
                "mean_best_epoch": float(np.mean(self.best_epochs))}
