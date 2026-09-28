"""Likelihoods, priors and scalar samplers for BoolMF (pure NumPy/SciPy)."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPS = 1e-12
LIKELIHOODS = ("noisy_or", "or_flip")


# --------------------------------------------------------------------------- likelihood
def presence_probability(likelihood, a, b, c):
    """P(x = 1 | c active components contain the feature).

    ``noisy_or``: a = transmission (detection) rate lambda, b = background rate epsilon,
    P = 1 - (1 - b) (1 - a)^c.  ``or_flip``: a = P(x = 1 | c >= 1), b = P(x = 1 | c = 0).
    """
    c = np.asarray(c)
    if likelihood == "noisy_or":
        p = 1.0 - (1.0 - b) * np.power(1.0 - a, c)
    else:
        p = np.where(c >= 1, a, b)
    return np.clip(p, _EPS, 1.0 - _EPS)


def loglik_tables(likelihood, a, b, cmax):
    """log P(x = 1 | c) and log P(x = 0 | c) for c = 0 .. cmax + 1."""
    p = presence_probability(likelihood, a, b, np.arange(cmax + 2))
    return np.log(p), np.log1p(-p)


def histogram_loglik(likelihood, a, b, H1, H0):
    """Log-likelihood of all observed entries from count histograms (exact)."""
    p = presence_probability(likelihood, a, b, np.arange(H1.size))
    return float(H1 @ np.log(p) + H0 @ np.log1p(-p))


# --------------------------------------------------------------------------- priors
@dataclass
class RatePrior:
    """Prior on a rate in (0, 1): a frozen SciPy distribution, or a float that fixes the rate."""

    dist: object = None
    fixed: float = None
    beta_ab: tuple = None

    @classmethod
    def from_param(cls, value, name):
        if value is None:
            return cls(dist=stats.beta(1.0, 1.0), beta_ab=(1.0, 1.0))
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            v = float(value)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} fixed at {v}; a fixed rate must lie in (0, 1).")
            return cls(fixed=v)
        if not (hasattr(value, "logpdf") and hasattr(value, "support")):
            raise TypeError(
                f"{name} must be None, a float in (0, 1), or a frozen scipy.stats distribution; "
                f"got {type(value).__name__}."
            )
        lo, hi = value.support()
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"{name} must have support inside [0, 1]; got [{lo}, {hi}].")
        return cls(dist=value, beta_ab=_beta_params(value))

    def logpdf(self, x):
        if self.beta_ab is not None:                 # fast path, constant dropped
            a, b = self.beta_ab
            return (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
        return float(self.dist.logpdf(x))


@dataclass
class PositivePrior:
    """Gamma prior (shape, rate) for the IBP concentration alpha."""

    shape: float = 1.0
    rate: float = 1.0

    @classmethod
    def from_param(cls, value, name):
        if value is None:
            return cls(1.0, 1.0)
        dname = getattr(getattr(value, "dist", None), "name", None)
        if dname != "gamma":
            raise ValueError(f"{name} must be None or a frozen scipy.stats.gamma distribution.")
        args, kw = value.args, value.kwds
        shape = args[0] if args else kw.get("a")
        loc = args[1] if len(args) > 1 else kw.get("loc", 0.0)
        scale = args[2] if len(args) > 2 else kw.get("scale", 1.0)
        if loc != 0:
            raise ValueError(f"{name}: loc must be 0.")
        return cls(float(shape), 1.0 / float(scale))


def beta_params_or_raise(value, name):
    """(a, b) of a Beta prior; None means Beta(1, 1)."""
    if value is None:
        return 1.0, 1.0
    ab = _beta_params(value)
    if ab is None:
        raise ValueError(f"{name} must be None or a frozen scipy.stats.beta distribution.")
    return ab


def _beta_params(dist):
    if getattr(getattr(dist, "dist", None), "name", None) != "beta":
        return None
    args, kw = dist.args, dist.kwds
    a = args[0] if args else kw.get("a")
    b = args[1] if len(args) > 1 else kw.get("b")
    loc = args[2] if len(args) > 2 else kw.get("loc", 0.0)
    scale = args[3] if len(args) > 3 else kw.get("scale", 1.0)
    if loc != 0 or scale != 1:
        return None
    return float(a), float(b)


# --------------------------------------------------------------------------- scalar samplers
def slice_sample_unit(logdens, x0, rng, w=1.0, max_steps=64):
    """One slice-sampling update (Neal 2003) of x in (lo, hi) inside (0, 1), on the logit scale.

    ``logdens(x)`` is the log target density in x; the logit Jacobian is added here.
    """

    def f(y):
        x = 1.0 / (1.0 + np.exp(-y))
        if not (_EPS < x < 1.0 - _EPS):
            return -np.inf
        return logdens(x) + np.log(x) + np.log1p(-x)

    y0 = np.log(x0) - np.log1p(-x0)
    fy0 = f(y0)
    level = fy0 + np.log(rng.random())
    left = y0 - w * rng.random()
    right = left + w
    j = int(max_steps * rng.random())
    k = max_steps - 1 - j
    while j > 0 and f(left) > level:
        left -= w
        j -= 1
    while k > 0 and f(right) > level:
        right += w
        k -= 1
    for _ in range(200):
        y1 = left + (right - left) * rng.random()
        if f(y1) > level:
            return 1.0 / (1.0 + np.exp(-y1))
        if y1 < y0:
            left = y1
        else:
            right = y1
    return x0


def truncated_beta(rng, a, b, lo=0.0, hi=1.0):
    """Draw from Beta(a, b) restricted to (lo, hi) by inverse CDF."""
    flo, fhi = stats.beta.cdf([lo, hi], a, b)
    if fhi - flo < 1e-300:
        return float(np.clip(stats.beta.mean(a, b), lo + _EPS, hi - _EPS))
    u = flo + (fhi - flo) * rng.random()
    return float(np.clip(stats.beta.ppf(u, a, b), lo + _EPS, hi - _EPS))


def update_rates(likelihood, a, b, H1, H0, prior_a, prior_b, rng):
    """Update the two rates given count histograms.

    ``or_flip`` with Beta priors uses the exact conjugate update (truncated so that the
    detection rate stays above the background rate); every other case uses slice sampling
    on the exact histogram log-likelihood.
    """
    if likelihood == "or_flip" and prior_a.fixed is None and prior_b.fixed is None \
            and prior_a.beta_ab and prior_b.beta_ab:
        n11, n01 = H1[1:].sum(), H0[1:].sum()
        n10, n00 = H1[0], H0[0]
        aa, ab = prior_a.beta_ab
        ba, bb = prior_b.beta_ab
        a = truncated_beta(rng, aa + n11, ab + n01, lo=b)
        b = truncated_beta(rng, ba + n10, bb + n00, hi=a)
        return a, b

    if prior_a.fixed is not None:
        a = prior_a.fixed
    else:
        lo_b = b if likelihood == "or_flip" else 0.0

        def la(x):
            if x <= lo_b:
                return -np.inf
            return histogram_loglik(likelihood, x, b, H1, H0) + prior_a.logpdf(x)

        a = slice_sample_unit(la, max(a, lo_b + 1e-9), rng)
    if prior_b.fixed is not None:
        b = prior_b.fixed
    else:
        hi_a = a if likelihood == "or_flip" else 1.0

        def lb(x):
            if x >= hi_a:
                return -np.inf
            return histogram_loglik(likelihood, a, x, H1, H0) + prior_b.logpdf(x)

        b = slice_sample_unit(lb, min(b, hi_a - 1e-9), rng)
    return a, b
