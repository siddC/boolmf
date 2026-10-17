"""Per-sample detection and background rates.

With ``detection_effects=("sample",)`` each sample i has its own detection rate
a_i = sigmoid(y_i), y_i ~ Normal(mu, sigma^2); likewise for the background rate with
``background_effects=("sample",)``. The population rate sigmoid(mu) carries the user's prior on
the rate (``detection_prior`` / ``background_prior``); sigma, the spread on the logit scale,
has a half-Cauchy(0, 1) prior. Everything is updated by exact MCMC: slice sampling
(Neal 2003) for each sample's logit rate given its count histogram, for mu and for sigma.

Under ``or_flip`` a sample's detection rate stays above its background rate, as the global rates
do.
"""

import math

import numpy as np
from numba import njit, prange

from .._model import slice_sample_unit
from .kernels import _next_uniform, _row_state

NOISY_OR, OR_FLIP = 0, 1
_EPS = 1e-12
SPREAD_SCALE = 1.0          # half-Cauchy scale of the logit-scale spread


def likelihood_code(likelihood):
    return NOISY_OR if likelihood == "noisy_or" else OR_FLIP


def table_rows(likelihood, a, b, cmax):
    """log P(x = 1 | c) and log P(x = 0 | c), c = 0 .. cmax + 1, one row per rate pair."""
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[:, None]
    c = np.arange(cmax + 2)[None, :]
    if likelihood == "noisy_or":
        p = 1.0 - (1.0 - b) * np.power(1.0 - a, c)
    else:
        p = np.where(c >= 1, a, b)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p), np.log1p(-p)


@njit(inline="always")
def _p1(lik, a, b, c):
    if lik == NOISY_OR:
        p = 1.0 - (1.0 - b) * (1.0 - a) ** c
    else:
        p = a if c >= 1 else b
    return min(max(p, _EPS), 1.0 - _EPS)


@njit(cache=True)
def _row_loglik(lik, a, b, h1, h0):
    s = 0.0
    for c in range(h1.size):
        if h1[c] == 0 and h0[c] == 0:
            continue
        p = _p1(lik, a, b, c)
        s += h1[c] * math.log(p) + h0[c] * math.log1p(-p)
    return s


@njit(cache=True)
def rows_loglik(lik, a, b, H1, H0):
    """Log-likelihood of all observed entries given per-sample rates a[i], b[i] and per-sample
    count histograms H1[i, c], H0[i, c]."""
    s = 0.0
    for i in range(H1.shape[0]):
        s += _row_loglik(lik, a[i], b[i], H1[i], H0[i])
    return s


@njit(inline="always")
def _sigmoid(y):
    return 1.0 / (1.0 + math.exp(-y))


@njit(cache=True)
def _target(lik, which, y, other, h1, h0, mu, sigma, lo, hi):
    r = _sigmoid(y)
    if not (lo < r < hi) or r <= _EPS or r >= 1.0 - _EPS:
        return -np.inf
    if which == 0:
        ll = _row_loglik(lik, r, other, h1, h0)
    else:
        ll = _row_loglik(lik, other, r, h1, h0)
    d = (y - mu) / sigma
    return ll - 0.5 * d * d


@njit(parallel=True, cache=True)
def slice_sample_row_logits(lik, which, y, other, H1, H0, mu, sigma, lo, hi, seed):
    """One slice-sampling update of every sample's logit rate y[i], in parallel.

    which = 0 updates detection rates (other = background rates), 1 background rates.
    The rate sigmoid(y[i]) is kept inside (lo[i], hi[i]).
    """
    n = y.size
    for i in prange(n):
        state = _row_state(seed, i)
        y0 = y[i]
        f0 = _target(lik, which, y0, other[i], H1[i], H0[i], mu, sigma, lo[i], hi[i])
        if not np.isfinite(f0):             # outside the constraint: restart at its midpoint
            mid = 0.5 * (max(lo[i], _EPS) + min(hi[i], 1.0 - _EPS))
            y0 = math.log(mid) - math.log1p(-mid)
            f0 = _target(lik, which, y0, other[i], H1[i], H0[i], mu, sigma, lo[i], hi[i])
            if not np.isfinite(f0):
                continue
        state, u = _next_uniform(state)
        level = f0 + math.log(max(u, 1e-300))
        state, u = _next_uniform(state)
        left = y0 - u
        right = left + 1.0
        steps = 32
        while steps > 0 and _target(lik, which, left, other[i], H1[i], H0[i], mu, sigma,
                                    lo[i], hi[i]) > level:
            left -= 1.0
            steps -= 1
        steps = 32
        while steps > 0 and _target(lik, which, right, other[i], H1[i], H0[i], mu, sigma,
                                    lo[i], hi[i]) > level:
            right += 1.0
            steps -= 1
        for _ in range(200):
            state, u = _next_uniform(state)
            y1 = left + (right - left) * u
            if _target(lik, which, y1, other[i], H1[i], H0[i], mu, sigma, lo[i], hi[i]) > level:
                y[i] = y1
                break
            if y1 < y0:
                left = y1
            else:
                right = y1


def slice_sample_real(logdens, x0, rng, w=1.0, max_steps=64):
    """One slice-sampling update of a real scalar."""
    level = logdens(x0) + np.log(rng.random())
    left = x0 - w * rng.random()
    right = left + w
    j = int(max_steps * rng.random())
    k = max_steps - 1 - j
    while j > 0 and logdens(left) > level:
        left -= w
        j -= 1
    while k > 0 and logdens(right) > level:
        right += w
        k -= 1
    for _ in range(200):
        x1 = left + (right - left) * rng.random()
        if logdens(x1) > level:
            return x1
        if x1 < x0:
            left = x1
        else:
            right = x1
    return x0


class SampleRates:
    """State and updates of one rate (detection or background) that varies by sample."""

    def __init__(self, n, rate0, prior):
        self.prior = prior                  # RatePrior on the population rate sigmoid(mu)
        self.y = np.full(n, np.log(rate0) - np.log1p(-rate0))
        self.mu = float(self.y[0])
        self.sigma = 0.5

    @property
    def rates(self):
        return 1.0 / (1.0 + np.exp(-self.y))

    @property
    def population_rate(self):
        return 1.0 / (1.0 + np.exp(-self.mu))

    def update(self, lik, which, other, H1, H0, lo, hi, rng):
        slice_sample_row_logits(lik, which, self.y, np.ascontiguousarray(other, float), H1, H0,
                                self.mu, self.sigma, lo, hi, np.uint64(rng.integers(0, 2**63 - 1)))
        y, sigma, prior = self.y, self.sigma, self.prior

        def log_mu(x):                      # x = sigmoid(mu), the population rate
            m = np.log(x) - np.log1p(-x)
            return prior.logpdf(x) - 0.5 * np.sum((y - m) ** 2) / sigma**2

        p = slice_sample_unit(log_mu, self.population_rate, rng)
        self.mu = float(np.log(p) - np.log1p(-p))
        ss = float(np.sum((y - self.mu) ** 2))
        n = y.size

        def log_logsigma(t):                # t = log sigma; half-Cauchy prior, Jacobian e^t
            s = np.exp(t)
            return -n * t - 0.5 * ss / s**2 - np.log1p((s / SPREAD_SCALE) ** 2) + t

        self.sigma = float(np.exp(slice_sample_real(log_logsigma, np.log(sigma), rng)))
