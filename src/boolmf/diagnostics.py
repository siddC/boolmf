"""MCMC convergence diagnostics: rank-normalized split R-hat and effective sample size.

Implements the recommendations of Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021),
"Rank-normalization, folding, and localization: an improved R-hat for assessing convergence
of MCMC", Bayesian Analysis 16(2).
"""

import numpy as np
from scipy import stats

__all__ = ["rhat", "ess", "segment_rhat", "geweke", "summarize_chains"]


def _as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("expected an array of shape (n_chains, n_draws)")
    return x


def _z_scale(x):
    """Rank-normalize all draws jointly to standard normal scores."""
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return stats.norm.ppf((r - 0.375) / (x.size + 0.25))


def _split(x):
    n = x.shape[1]
    half = n // 2
    if half < 2:
        return x
    return np.concatenate([x[:, -2 * half:-half], x[:, -half:]], axis=0)


def _rhat_basic(x):
    m, n = x.shape
    if n < 2 or m < 2:
        return np.nan
    means = x.mean(axis=1)
    within = x.var(axis=1, ddof=1).mean()
    between = n * means.var(ddof=1)
    if within <= 0:
        return 1.0 if between <= 0 else np.inf
    var_hat = (n - 1) / n * within + between / n
    return float(np.sqrt(var_hat / within))


def rhat(x):
    """Rank-normalized split R-hat: max of the bulk and the folded (tail) versions.

    Parameters
    ----------
    x : array-like of shape (n_chains, n_draws)

    Returns
    -------
    float
        Values near 1 mean the chains agree; 1.01 is the recommended cut-off.
    """
    x = _as_2d(x)
    x = x[:, ~np.all(np.isnan(x), axis=0)]
    if x.size == 0 or np.all(x == x.flat[0]):
        return 1.0
    xs = _split(x)
    bulk = _rhat_basic(_z_scale(xs))
    folded = np.abs(xs - np.median(xs))
    tail = _rhat_basic(_z_scale(folded)) if not np.all(folded == folded.flat[0]) else 1.0
    return float(np.nanmax([bulk, tail]))


def segment_rhat(x, n_segments=4):
    """R-hat of one chain cut into ``n_segments`` consecutive pieces (detects drift)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    L = x.size // n_segments
    if L < 2:
        return np.inf
    return rhat(x[-L * n_segments:].reshape(n_segments, L))


def geweke(x, first=0.1, last=0.5):
    """Geweke z-score comparing the mean of the first 10% and the last 50% of one chain.

    The variance of each mean uses its effective sample size, so slowly mixing but
    stationary traces are not mistaken for drift. |z| <= 2 indicates no trend.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    a, b = x[: max(2, int(first * n))], x[int((1 - last) * n):]
    if a.size < 4 or b.size < 4:
        return np.inf
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return 0.0 if a.mean() == b.mean() else np.inf
    se2 = va / max(ess(a[None, :]), 1.0) + vb / max(ess(b[None, :]), 1.0)
    return float(abs(a.mean() - b.mean()) / np.sqrt(se2))


def _autocov(x):
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=size, axis=-1)
    acov = np.fft.irfft(f * np.conjugate(f), n=size, axis=-1)[..., :n]
    return acov / n


def ess(x):
    """Bulk effective sample size (rank-normalized, Geyer's initial monotone sequence)."""
    x = _as_2d(x)
    x = x[:, ~np.all(np.isnan(x), axis=0)]
    m, n = x.shape
    if n < 4:
        return float(m * n)
    if np.all(x == x.flat[0]):
        return float(m * n)
    xs = _split(x) if n >= 8 else x
    z = _z_scale(xs)
    m, n = z.shape
    acov = _autocov(z)
    chain_mean = z.mean(axis=1)
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += chain_mean.var(ddof=1)
    if var_plus <= 0:
        return float(m * n)
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer's initial positive and monotone sequence
    t = 0
    s = 0.0
    prev = np.inf
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        p = min(p, prev)
        s += p
        prev = p
        t += 2
    tau = -1.0 + 2.0 * s
    tau = max(tau, 1.0 / np.log10(m * n)) if m * n > 1 else 1.0
    return float(m * n / tau)


def summarize_chains(traces):
    """R-hat and ESS for each monitored scalar.

    Parameters
    ----------
    traces : dict of name -> array of shape (n_chains, n_draws)

    Returns
    -------
    dict of name -> {"rhat": float, "ess": float}
    """
    return {k: {"rhat": rhat(v), "ess": ess(v)} for k, v in traces.items()}
