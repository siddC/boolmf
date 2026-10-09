"""``likelihood_power``: the coarsened posterior, likelihood^zeta x prior (Miller & Dunson 2019).

The exact check enumerates the tempered posterior of a problem small enough to sum over (2 rows,
2 features, one entry missing), as in ``test_collapsed``, and compares it with the estimator's
collapsed IBP sampler run at zeta = 0.4 with fixed rates, alpha and membership probability.
"""

import itertools
import math
import warnings

import numpy as np
import pytest

from boolmf import BayesianBooleanMF
from boolmf._model import loglik_tables
from boolmf._sampler.engines import CountEngine
from boolmf._sampler.kernels import counts_from_state

from .conftest import FAST

V = np.array([[1, 0], [1, -1]], np.int8)
ALPHA, P, KMAX = 1.0, 0.4, 8


def _exact(zeta, likelihood):
    """Distributions of C (counts capped at 3) and of the number of components with members."""
    T1, T0 = loglik_tables(likelihood, 0.8, 0.1, 30)
    T1, T0 = zeta * T1, zeta * T0
    n_rows, n_feat = V.shape
    carriers = [(1, 0), (0, 1), (1, 1)]
    members = list(itertools.product([0, 1], repeat=n_feat))
    types = [(h, u) for h in carriers for u in members]
    observed = V >= 0
    dist, kdist = {}, np.zeros(KMAX + 1)
    for K in range(KMAX + 1):
        for combo in itertools.combinations_with_replacement(range(len(types)), K):
            n = np.bincount(combo, minlength=len(types))
            lp = K * math.log(ALPHA) - sum(math.lgamma(c + 1) for c in n)
            C = np.zeros(V.shape, int)
            for t in combo:
                h, u = types[t]
                m = sum(h)
                lp += (math.lgamma(n_rows - m + 1) + math.lgamma(m) - math.lgamma(n_rows + 1)
                       + sum(u) * math.log(P) + (n_feat - sum(u)) * math.log1p(-P))
                C += np.outer(h, u)
            lp += np.where(V == 1, T1[C], T0[C])[observed].sum()
            w = math.exp(lp)
            key = tuple(np.minimum(C, 3).ravel())
            dist[key] = dist.get(key, 0.0) + w
            kdist[sum(1 for t in combo if any(types[t][1]))] += w
    z = kdist.sum()
    return {k: v / z for k, v in dist.items()}, kdist / z


@pytest.mark.parametrize("likelihood", ["or_flip", "noisy_or"])
def test_estimator_targets_exact_tempered_posterior(likelihood):
    zeta, n_draws = 0.4, 20000
    exact, exact_k = _exact(zeta, likelihood)
    full, full_k = _exact(1.0, likelihood)
    X = np.where(V < 0, np.nan, V).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BayesianBooleanMF(likelihood=likelihood, likelihood_power=zeta, births="enumerate",
                              membership_level="shared", split_merge=False, membership_prior=P,
                              alpha_prior=ALPHA, detection_prior=0.8, background_prior=0.1,
                              max_components=24, init="empty", min_support=1, n_chains=1,
                              burn_in=500, n_draws=n_draws, thin=1, max_sweeps=n_draws + 500,
                              store_draws=True, random_state=0).fit(X)
    k = np.asarray(m.n_components_draws_).ravel()[-n_draws:]
    k_freq = np.bincount(k, minlength=KMAX + 1)[:KMAX + 1] / k.size
    assert np.abs(k_freq - exact_k).max() < 0.015
    assert np.abs(exact_k - full_k).max() > 0.05          # the power changes the target
    hist = {}
    for d in m._draws_:
        Z = np.unpackbits(d["Z"], axis=0, count=2).astype(int)
        U = np.unpackbits(d["U"], axis=0, count=2).astype(int)
        key = tuple(np.minimum(Z @ U.T, 3).ravel())
        hist[key] = hist.get(key, 0) + 1
    top = sorted(exact, key=lambda s: -exact[s])[:8]
    assert max(abs(hist.get(s, 0) / len(m._draws_) - exact[s]) for s in top) < 0.012
    assert max(abs(full.get(s, 0) - exact[s]) for s in top) > 0.03


def test_rate_updates_use_tempered_counts():
    # or_flip, global rates, Beta(1, 1) priors: given Z and U the detection rate is drawn from
    # Beta(1 + zeta n11, 1 + zeta n01) truncated above the background rate
    rng = np.random.default_rng(0)
    Z = (rng.random((60, 3)) < 0.4).astype(np.int8)
    U = (rng.random((40, 3)) < 0.3).astype(np.int8)
    truth = counts_from_state(Z, U) > 0
    V = np.where(rng.random(truth.shape) < 0.1, ~truth, truth).astype(np.int8)
    n11 = int((V == 1)[truth].sum())
    n01 = int((V == 0)[truth].sum())
    m = BayesianBooleanMF(likelihood_power=0.25)
    cfg = _config(m, V)
    eng = CountEngine(cfg, V, Z, U, rng)
    draws = []
    for _ in range(4000):
        a, _, ll = eng.update_rates(V, Z, U, rng)
        draws.append(a)
    draws = np.array(draws)
    A, B = 1 + 0.25 * n11, 1 + 0.25 * n01
    assert abs(draws.mean() - A / (A + B)) < 0.003
    sd = math.sqrt(A * B / ((A + B) ** 2 * (A + B + 1)))
    assert abs(draws.std() / sd - 1) < 0.08
    # the reported log-likelihood is the model's own, not tempered
    T1, T0 = loglik_tables("or_flip", eng.a, eng.b, 3)
    C = np.minimum(counts_from_state(Z, U), 1)
    assert ll == pytest.approx(float(np.where(V == 1, T1[C], T0[C]).sum()))


def test_per_sample_rates_widen_under_coarsening():
    # per-sample detection rates given Z and U: with zeta = 0.1 each sample's rate rests on a
    # tenth of its entries, so its posterior spread is wider than with zeta = 1 (by less than
    # sqrt(10): every sample has the same true rate, so the learned spread pulls them together)
    rng = np.random.default_rng(1)
    Z = (rng.random((50, 3)) < 0.4).astype(np.int8)
    U = (rng.random((600, 3)) < 0.3).astype(np.int8)
    truth = counts_from_state(Z, U) > 0
    V = np.where(rng.random(truth.shape) < 0.1, ~truth, truth).astype(np.int8)
    sd = {}
    for zeta in (1.0, 0.1):
        cfg = _config(BayesianBooleanMF(likelihood_power=zeta), V, detection_effects=("sample",))
        eng = CountEngine(cfg, V, Z, U, np.random.default_rng(0))
        draws = []
        for i in range(1500):
            eng.update_rates(V, Z, U, rng)
            if i >= 300:
                draws.append(eng.det_s[0].rates.copy())
        sd[zeta] = np.std(draws, axis=0).mean()
    assert sd[0.1] > 1.25 * sd[1.0]          # 1.5 here; 0.85 if the counts are not tempered


def _config(estimator, V, **kw):
    from boolmf._model import PositivePrior, RatePrior
    from boolmf._sampler.chain import ChainConfig

    return ChainConfig(likelihood="or_flip", n_slots=3, anchor_members=[], anchor_learned=[],
                       nonparametric=False, prior_a=RatePrior.from_param(None, "a"),
                       prior_b=RatePrior.from_param(None, "b"), membership_ab=(1.0, 1.0),
                       alpha_prior=PositivePrior.from_param(None, "alpha"), burn_in=0,
                       max_sweeps=1, n_draws=1, thin=1, rhat_threshold=1.1, n_init=0,
                       likelihood_power=float(estimator.likelihood_power), **kw)


def test_power_one_is_the_usual_posterior(small_data):
    X, _ = small_data
    a = BayesianBooleanMF(**FAST).fit(X)
    b = BayesianBooleanMF(likelihood_power=1.0, **FAST).fit(X)
    np.testing.assert_array_equal(a.log_likelihood_trace_, b.log_likelihood_trace_)
    np.testing.assert_array_equal(a.components_, b.components_)


def test_coarsening_needs_more_support(small_data):
    # small_data has three planted components; with zeta small enough the prior outweighs
    # the evidence for them and fewer components are kept
    X, _ = small_data
    full = BayesianBooleanMF(**FAST).fit(X)
    coarse = BayesianBooleanMF(likelihood_power=0.002, **FAST).fit(X)
    assert full.n_components_ == 3
    assert coarse.n_components_ < full.n_components_


def test_likelihood_power_validation(small_data):
    X, _ = small_data
    for bad in (0.0, 1.5, -0.1, True, "half"):
        with pytest.raises(ValueError, match="likelihood_power"):
            BayesianBooleanMF(likelihood_power=bad).fit(X)
    with pytest.raises(NotImplementedError, match="likelihood_power"):
        BayesianBooleanMF(likelihood="noisy_or", detection_effects=("component",),
                          likelihood_power=0.5).fit(X)
