"""Rates per feature group.

* The Gibbs kernels and split-merge moves must keep the exact posterior when entries read their
  likelihood from a (sample, group) table row: checked on a problem small enough to enumerate,
  with two groups whose per-sample rates differ.
* A fit with groups recovers group-specific rates; one group gives the same fit as no groups.
"""

import numpy as np
import pytest
from scipy import stats

from boolmf import BayesianBooleanMF
from boolmf._sampler.chain import _csr
from boolmf._sampler.kernels import counts_from_state, update_activations, update_memberships
from boolmf._sampler.population import _delta_ll
from boolmf._sampler.rates import table_rows
from boolmf._sampler.splitmerge import _log_beta_bernoulli, split_merge_moves

V = np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], np.int8)
N, F = V.shape
K = 2
FG = np.array([0, 1, 1], np.int64)                 # feature 0 in group 0, features 1-2 in group 1
G = 2
BITS = N * K + F * K
# per-sample rates, different in the two groups: rows (i * G + g)
_T = [table_rows("or_flip", [0.85, 0.8, 0.9], [0.1, 0.2, 0.05], K),
      table_rows("or_flip", [0.6, 0.7, 0.65], [0.3, 0.25, 0.35], K)]
T1 = np.stack([t[0] for t in _T], axis=1).reshape(N * G, K + 2)
T0 = np.stack([t[1] for t in _T], axis=1).reshape(N * G, K + 2)
ROW = np.arange(N)[:, None] * G + FG[None, :]      # table row of every entry
LOGIT_PI = np.array([[0.3, -0.4]])
LOGIT_RHO = np.array([[-0.2, 0.5]])
ZPRIOR, UPRIOR = (0.7, 1.0), (1.0, 1.5)


def _state(code):
    zbits, ubits = code >> (F * K), code & ((1 << (F * K)) - 1)
    Z = np.array([(zbits >> t) & 1 for t in range(N * K)], np.int8).reshape(N, K)
    U = np.array([(ubits >> t) & 1 for t in range(F * K)], np.int8).reshape(F, K)
    return Z, U


def _code(Z, U):
    return (sum(int(v) << t for t, v in enumerate(Z.ravel())) << (F * K)) | \
        sum(int(v) << t for t, v in enumerate(U.ravel()))


def _loglik(Z, U):
    C = counts_from_state(Z, U).astype(int)
    return np.where(V == 1, T1[ROW, C], T0[ROW, C])[V >= 0].sum()


def _posterior(collapsed):
    logp = np.empty(1 << BITS)
    for code in range(logp.size):
        Z, U = _state(code)
        lp = _loglik(Z, U)
        if collapsed:                                # Beta-Bernoulli (split-merge target)
            for k in range(K):
                lp += _log_beta_bernoulli(int(Z[:, k].sum()), *ZPRIOR, N)
                lp += _log_beta_bernoulli(int(U[:, k].sum()), *UPRIOR, F)
        else:                                        # fixed pi, rho (Gibbs target)
            lp += (Z * LOGIT_PI).sum() + (U * LOGIT_RHO).sum()
        logp[code] = lp
    p = np.exp(logp - logp.max())
    return p / p.sum()


def _chi2_ok(p, counts):
    expected = p * counts.sum()
    big = expected >= 5
    chi2 = ((counts[big] - expected[big]) ** 2 / expected[big]).sum()
    return stats.chi2.sf(chi2, big.sum() - 1) > 1e-3


def test_gibbs_kernels_with_groups_keep_the_exact_posterior():
    p = _posterior(collapsed=False)
    rng = np.random.default_rng(0)
    counts = np.zeros(p.size)
    every = np.ones(K, bool)
    for code in rng.choice(p.size, 20000, p=p):
        Z, U = _state(code)
        C = counts_from_state(Z, U)
        act_ptr, act_idx = _csr(Z)
        update_memberships(V, U, C, T1, T0, LOGIT_RHO, act_ptr, act_idx, every,
                           np.uint64(rng.integers(2**63)), False, FG)
        mem_ptr, mem_idx = _csr(U)
        update_activations(V, Z, C, T1, T0, LOGIT_PI, mem_ptr, mem_idx, every,
                           np.uint64(rng.integers(2**63)), False, FG)
        np.testing.assert_array_equal(C, counts_from_state(Z, U))
        counts[_code(Z, U)] += 1
    assert _chi2_ok(p, counts)


def test_split_merge_with_groups_keeps_the_exact_posterior():
    p = _posterior(collapsed=True)
    rng = np.random.default_rng(1)
    counts = np.zeros(p.size)
    moves = np.zeros((2, 5), np.int64)
    for code in rng.choice(p.size, 20000, p=p):
        Z, U = _state(code)
        C = counts_from_state(Z, U)
        split_merge_moves(V, Z, U, C, T1, T0, np.ones(K, bool), ZPRIOR, UPRIOR, 2, 2, rng, moves,
                          FG)
        np.testing.assert_array_equal(C, counts_from_state(Z, U))
        counts[_code(Z, U)] += 1
    assert (moves[1, :3] > 40).all()
    assert _chi2_ok(p, counts)


def test_population_delta_uses_the_group_rows():
    from types import SimpleNamespace

    rng = np.random.default_rng(2)
    for _ in range(50):
        Z, U = _state(int(rng.integers(1 << BITS)))
        eng = SimpleNamespace(C=counts_from_state(Z, U), T1=T1, T0=T0, fg=FG, G=G)
        member = SimpleNamespace(engine=eng)
        z = (rng.random(N) < 0.5).astype(np.int8)
        u = (rng.random(F) < 0.5).astype(np.int8)
        rows, cols = np.flatnonzero(z), np.flatnonzero(u)
        Z2, U2 = np.c_[Z, z], np.c_[U, u]
        assert _delta_ll(V, member, rows, cols, 1) == pytest.approx(_loglik(Z2, U2) - _loglik(Z, U))


def _two_group_data(seed=0):
    from boolmf.datasets import make_boolean_factors

    X, truth = make_boolean_factors(150, 80, 3, prevalence=(0.3, 0.5), membership=(0.2, 0.4),
                                    detection=1.0, background=0.0, random_state=seed,
                                    return_truth=True)
    rng = np.random.default_rng(seed)
    groups = np.repeat(["clean", "noisy"], 40)
    a = np.where(groups == "clean", 0.97, 0.75)
    b = np.where(groups == "clean", 0.01, 0.12)
    p = np.where(X == 1, a, b)
    return (rng.random(X.shape) < p).astype(float), groups


FAST = dict(n_chains=2, max_sweeps=300, burn_in=150, n_draws=50, thin=3, random_state=0)


def test_group_rates_are_recovered():
    X, groups = _two_group_data()
    m = BayesianBooleanMF(n_components=3, feature_groups=groups, **FAST).fit(X)
    assert list(m.feature_groups_) == ["clean", "noisy"]
    np.testing.assert_allclose(m.detection_rate_per_group_, [0.97, 0.75], atol=0.04)
    np.testing.assert_allclose(m.background_rate_per_group_, [0.01, 0.12], atol=0.03)
    plain = BayesianBooleanMF(n_components=3, **FAST).fit(X)
    assert not hasattr(plain, "detection_rate_per_group_") and plain.feature_groups_ is None
    # the grouped model predicts the noisy group's absences better
    assert m.score(X) > plain.score(X)
    P = m.predictive_probability(X[:20])
    assert P.shape == (20, X.shape[1]) and P.min() >= 0 and P.max() <= 1


def test_per_sample_rates_per_group_and_single_group_equals_none():
    X, groups = _two_group_data(1)
    m = BayesianBooleanMF(n_components=3, feature_groups=groups, detection_effects=("sample",),
                          background_effects=("sample",), **FAST).fit(X)
    assert m.detection_rate_per_sample_.shape == (X.shape[0], 2)
    assert m.detection_rate_per_sample_interval_.shape == (X.shape[0], 2, 2)
    assert (m.detection_rate_per_sample_[:, 0] > m.detection_rate_per_sample_[:, 1]).mean() > 0.9
    a = BayesianBooleanMF(n_components=3, feature_groups=np.zeros(X.shape[1]), **FAST).fit(X)
    b = BayesianBooleanMF(n_components=3, **FAST).fit(X)
    np.testing.assert_array_equal(a.log_likelihood_trace_, b.log_likelihood_trace_)
    np.testing.assert_array_equal(a.components_, b.components_)


def test_groups_with_population_moves_run(small_data):
    X, _ = small_data
    groups = np.arange(X.shape[1]) % 2
    m = BayesianBooleanMF(n_chains=2, max_sweeps=60, burn_in=30, n_draws=10, thin=1,
                          random_state=0, population_moves=5, feature_groups=groups).fit(X)
    assert m.detection_rate_per_group_.shape == (2,)


@pytest.mark.parametrize("params, error, match", [
    (dict(feature_groups=[0, 1]), ValueError, "one label per feature"),
    (dict(feature_groups="x", births="enumerate", split_merge=False, membership_level="shared"),
     ValueError, "births='slots'"),
    (dict(feature_groups="x", likelihood="noisy_or", detection_effects=("component",)),
     NotImplementedError, "not implemented"),
])
def test_feature_group_validation(small_data, params, error, match):
    X, _ = small_data
    if isinstance(params.get("feature_groups"), str):
        params = {**params, "feature_groups": np.zeros(X.shape[1])}
    with pytest.raises(error, match=match):
        BayesianBooleanMF(n_chains=1, max_sweeps=20, burn_in=10, n_draws=5, **params).fit(X)
