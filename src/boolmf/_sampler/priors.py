"""Priors on the two binary factor matrices.

Each side (activations Z, rows = samples; memberships U, rows = features) has one of these
models for P(entry = 1):

* ``"ibp"``        truncated Indian buffet process: pi_k ~ Beta(alpha / K, 1) per slot
  (activations only; memberships reach it by transposing the problem);
* ``"component"``  one rate per component, rate_k ~ Beta(a, b);
* ``"shared"``     one rate for every entry of the side, rate ~ Beta(a, b);
* ``"row"``        one rate per row (per sample for Z, per feature for U), rate_r ~ Beta(a, b).

A rate can also be fixed (``fixed``), in which case it is never updated. ``logits()`` returns the
prior log-odds as a 2-D array broadcast by the kernels: (1, K), (1, 1) or (n_rows, 1).
"""

import numpy as np

MODELS = ("ibp", "component", "shared", "row")


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


class SidePrior:
    def __init__(self, model, n_rows, K, beta_ab=(1.0, 1.0), fixed=None, alpha_prior=None,
                 fixed_cols=None):
        if model not in MODELS:
            raise ValueError(f"unknown prior model {model!r}")
        self.model, self.n, self.K = model, n_rows, K
        self.a, self.b = beta_ab
        self.fixed = fixed
        self.alpha_prior = alpha_prior             # PositivePrior (shape, rate) or fixed float
        self.fixed_cols = np.zeros(K, bool) if fixed_cols is None else fixed_cols
        start = 0.5 if fixed is None else fixed
        if model == "row":
            self.rate = np.full(n_rows, start)
        elif model in ("ibp", "component"):
            self.rate = np.full(K, start)
        else:
            self.rate = np.array([start])
        self.alpha = float(alpha_prior) if isinstance(alpha_prior, float) else 1.0

    @property
    def collapsible(self):
        """Beta–Bernoulli per column (what split–merge moves integrate out)."""
        return self.model in ("ibp", "component") and self.fixed is None

    def init_from(self, B):
        """Start the rates at the empirical frequencies of the initial state B (rows x K)."""
        if self.fixed is not None:
            return
        if self.model in ("ibp", "component"):
            self.rate = np.clip(B.mean(0), 1e-4, 1 - 1e-4).astype(float)
        elif self.model == "shared":
            self.rate = np.array([float(np.clip(B.mean(), 1e-4, 1 - 1e-4))])
        else:
            self.rate = np.clip(B.mean(1), 1e-4, 1 - 1e-4).astype(float)

    def update(self, B, cols, rng):
        """Draw the rates from their conditionals given B (rows x K); ``cols`` marks the columns
        (components) that exist for this side's rates."""
        if self.fixed is not None:
            return
        n = self.n
        if self.model == "ibp":
            Kf = max(1, int(cols.sum()))
            m = B[:, cols].sum(0)
            pi = np.clip(rng.beta(self.alpha / Kf + m, 1.0 + n - m), 1e-300, 1 - 1e-12)
            self.rate = np.ones(self.K)
            self.rate[cols] = pi
            ap = self.alpha_prior
            if not isinstance(ap, float):
                rate = ap.rate - np.log(pi).sum() / Kf
                self.alpha = rng.gamma(ap.shape + Kf, 1.0 / rate)
            return
        if self.model == "component":
            m = B.sum(0)
            self.rate = rng.beta(self.a + m, self.b + n - m).astype(float)
            return
        Bc = B[:, cols]
        if self.model == "shared":
            s, tot = float(Bc.sum()), float(Bc.size)
            self.rate = np.array([rng.beta(self.a + s, self.b + tot - s)])
        else:
            s = Bc.sum(1)
            self.rate = rng.beta(self.a + s, self.b + Bc.shape[1] - s).astype(float)

    def logits(self):
        lg = _logit(self.rate)
        if self.model in ("ibp", "component"):
            out = lg[None, :].copy()
            out[0, self.fixed_cols] = 50.0          # anchors: always active
            return out
        if self.model == "shared":
            return lg.reshape(1, 1)
        return lg[:, None]

    def collapsed_ab(self, Kf):
        """(a, b) of the Beta prior per column, for split–merge moves."""
        if self.model == "ibp":
            return self.alpha / Kf, 1.0
        return self.a, self.b

    def mean_rate(self):
        """A single activation probability for a new row (used when projecting new samples)."""
        return float(np.mean(self.rate))
