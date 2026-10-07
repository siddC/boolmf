"""Population moves (transplant, delete, crossover) must leave every chain's posterior exact.

Two chains with different rates and Indian-buffet priors share a problem small enough to
enumerate (3 samples, 3 features, two free slots, one missing entry). Pairs of states drawn
from the product of the two exact posteriors must still follow it after population moves:
each chain's marginal, and the joint distribution of the number of used slots.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from boolmf import BayesianBooleanMF
from boolmf._model import loglik_tables
from boolmf._sampler import engines
from boolmf._sampler.chain import Member
from boolmf._sampler.kernels import counts_from_state
from boolmf._sampler.population import (
    N_MOVES,
    PopulationConfig,
    crossover,
    data_regions,
    transplant_or_delete,
)
from boolmf._sampler.rates import table_rows
from boolmf._sampler.splitmerge import _log_beta_bernoulli

V = np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], np.int8)
N, F = V.shape
K = 2
BITS = N * K + F * K
FREE = np.ones(K, bool)
REGIONS = {"samples": [np.array([0, 1]), np.array([1, 2]), np.array([0, 1, 2])],
           "features": [np.array([0, 2]), np.array([1, 2]), np.array([0, 1, 2])]}
# chain 0: global rates; chain 1: per-sample rates (one table row per sample), other priors
TABLES = [tuple(t[None, :] for t in loglik_tables("or_flip", 0.8, 0.2, K)),
          table_rows("or_flip", [0.9, 0.7, 0.85], [0.1, 0.25, 0.05], K)]
ZPRIORS = [(0.3, 1.0), (1.5, 1.0)]
UPRIORS = [(1.0, 2.0), (0.7, 1.0)]


def _state(code):
    zbits, ubits = code >> (F * K), code & ((1 << (F * K)) - 1)
    Z = np.array([(zbits >> t) & 1 for t in range(N * K)], np.int8).reshape(N, K)
    U = np.array([(ubits >> t) & 1 for t in range(F * K)], np.int8).reshape(F, K)
    return Z, U


def _code(Z, U):
    zbits = sum(int(v) << t for t, v in enumerate(Z.ravel()))
    ubits = sum(int(v) << t for t, v in enumerate(U.ravel()))
    return (zbits << (F * K)) | ubits


def _posterior(c):
    T1, T0 = TABLES[c]
    rows = np.arange(N)[:, None] if T1.shape[0] > 1 else np.zeros((N, 1), int)
    logp = np.empty(1 << BITS)
    for code in range(logp.size):
        Z, U = _state(code)
        C = counts_from_state(Z, U).astype(int)
        lp = np.where(V == 1, T1[rows, C], T0[rows, C])[V >= 0].sum()
        for k in range(K):
            lp += _log_beta_bernoulli(int(Z[:, k].sum()), *ZPRIORS[c], N)
            lp += _log_beta_bernoulli(int(U[:, k].sum()), *UPRIORS[c], F)
        logp[code] = lp
    p = np.exp(logp - logp.max())
    return p / p.sum()


def _member(c, code):
    Z, U = _state(code)
    eng = SimpleNamespace(C=counts_from_state(Z, U), T1=TABLES[c][0], T0=TABLES[c][1])
    return Member(Z, U, eng, FREE, ZPRIORS[c], UPRIORS[c])


def _used(code):
    Z, U = _state(code)
    return int((Z.any(0) & U.any(0)).sum())


def test_population_moves_preserve_the_product_of_exact_posteriors():
    p = [_posterior(0), _posterior(1)]
    rng = np.random.default_rng(3)
    n_draws = 30000
    counts = [np.zeros(p[0].size), np.zeros(p[1].size)]
    joint = np.zeros((K + 1, K + 1))
    moves = np.zeros((2, N_MOVES), np.int64)
    draws = zip(rng.choice(p[0].size, n_draws, p=p[0]), rng.choice(p[1].size, n_draws, p=p[1]))
    for c0, c1 in draws:
        m = [_member(0, c0), _member(1, c1)]
        for _ in range(3):
            transplant_or_delete(V, m[0], m[1], rng, moves)
            transplant_or_delete(V, m[1], m[0], rng, moves)
            kind = ("samples", "features")[rng.integers(2)]
            region = REGIONS[kind][rng.integers(3)]
            a, b = (0, 1) if rng.random() < 0.5 else (1, 0)
            crossover(V, m[a], m[b], kind, region, rng, moves)
        codes = []
        for c in (0, 1):
            np.testing.assert_array_equal(m[c].engine.C, counts_from_state(m[c].Z, m[c].U))
            codes.append(_code(m[c].Z, m[c].U))
            counts[c][codes[-1]] += 1
        joint[_used(codes[0]), _used(codes[1])] += 1
    assert (moves[1] > 200).all()                    # every move type is accepted often

    used = np.array([_used(code) for code in range(p[0].size)])
    exp_joint = np.outer([p[0][used == u].sum() for u in range(K + 1)],
                         [p[1][used == u].sum() for u in range(K + 1)]) * n_draws
    ok = exp_joint >= 5
    chi2 = ((joint[ok] - exp_joint[ok]) ** 2 / exp_joint[ok]).sum()
    assert stats.chi2.sf(chi2, ok.sum() - 1) > 1e-3
    for c in (0, 1):
        expected = p[c] * n_draws
        big = expected >= 5
        chi2 = ((counts[c][big] - expected[big]) ** 2 / expected[big]).sum()
        assert stats.chi2.sf(chi2, big.sum() - 1) > 1e-3


def test_data_regions_respect_the_size_bounds():
    rng = np.random.default_rng(0)
    X = (rng.random((40, 30)) < 0.3).astype(np.int8)
    pcfg = PopulationConfig(region_min_size=3, region_max_fraction=0.5)
    regions = data_regions(X, pcfg)
    assert set(regions) == {"samples", "features"}
    for kind, total in (("samples", 40), ("features", 30)):
        sizes = [r.size for r in regions[kind]]
        assert min(sizes) >= 3 and max(sizes) <= total / 2
        assert all(np.unique(r).size == r.size for r in regions[kind])
    assert "features" not in data_regions(X, PopulationConfig(max_region_items=10))


POP = dict(n_chains=3, max_sweeps=120, burn_in=60, n_draws=30, thin=2, random_state=0,
           population_moves=5)


def test_population_fit_is_reproducible_and_reports_moves(small_data):
    X, truth = small_data
    a = BayesianBooleanMF(**POP).fit(X)
    b = BayesianBooleanMF(**POP).fit(X)
    np.testing.assert_array_equal(a.log_likelihood_trace_, b.log_likelihood_trace_)
    np.testing.assert_array_equal(a.components_, b.components_)
    assert a.population_acceptance_ == b.population_acceptance_
    assert set(a.population_acceptance_) == {"transplant", "delete", "crossover_samples",
                                             "crossover_features"}
    assert a.population_acceptance_["transplant"][0] > 0
    assert a.n_components_ >= 1
    assert BayesianBooleanMF(n_chains=2, max_sweeps=60, burn_in=30, n_draws=10, thin=1,
                             random_state=0).fit(X).population_acceptance_ is None


class Interrupt(Exception):
    pass


def test_population_resumes_from_its_checkpoint(small_data, tmp_path, monkeypatch):
    X, _ = small_data
    params = {**POP, "checkpoint_every": 20}
    clean = BayesianBooleanMF(checkpoint_dir=tmp_path / "clean", **params).fit(X)
    assert os.listdir(tmp_path / "clean") == []

    calls = {"n": 0}
    original = engines.CountEngine.update_rates

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3 * 47 + 2:                     # sweep 47, third chain
            raise Interrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(engines.CountEngine, "update_rates", flaky)
    with pytest.raises(Interrupt):
        BayesianBooleanMF(checkpoint_dir=tmp_path / "run", **params).fit(X)
    assert os.listdir(tmp_path / "run") == ["population.pkl"]
    monkeypatch.setattr(engines.CountEngine, "update_rates", original)

    resumed = BayesianBooleanMF(checkpoint_dir=tmp_path / "run", **params).fit(X)
    np.testing.assert_array_equal(resumed.log_likelihood_trace_, clean.log_likelihood_trace_)
    np.testing.assert_array_equal(resumed.components_, clean.components_)
    assert resumed.population_acceptance_ == clean.population_acceptance_
    assert os.listdir(tmp_path / "run") == []


def test_per_chain_starts(small_data):
    X, truth = small_data
    H, W = truth["members"].astype(bool), truth["activations"].astype(bool)
    starts = [(H, W), (H[:2], W[:, :2])]
    m = BayesianBooleanMF(n_chains=2, init=starts, max_sweeps=40, burn_in=20, n_draws=10,
                          thin=1, random_state=0).fit(X)
    assert m.n_components_ >= 1
    with pytest.raises(ValueError, match="one \\(members, activations\\) tuple per chain"):
        BayesianBooleanMF(n_chains=3, init=starts).fit(X)


@pytest.mark.parametrize("params, error, match", [
    (dict(population_moves=-1), ValueError, "population_moves must be an int"),
    (dict(population_moves=5, n_chains=1), ValueError, "n_chains >= 2"),
    (dict(population_moves=5, births="enumerate", split_merge=False,
          membership_level="shared"), ValueError, "births='slots'"),
    (dict(population_moves=5, likelihood="noisy_or", detection_effects=("component",)),
     NotImplementedError, "not implemented"),
    (dict(population_moves=5, membership_level="shared", split_merge=False), ValueError,
     "Beta membership rate per component"),
    (dict(population_moves=5, population_params={"n_swaps": 3}), ValueError, "does not take"),
    (dict(population_moves=5, population_params={"region_max_fraction": 0}), ValueError,
     "region_max_fraction"),
    (dict(population_params=[1]), ValueError, "population_params must be a dict"),
])
def test_population_validation(small_data, params, error, match):
    X, _ = small_data
    with pytest.raises(error, match=match):
        BayesianBooleanMF(**{"n_chains": 2, **params}).fit(X)
