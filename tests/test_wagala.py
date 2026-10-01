"""BetaMixture priors, Asso starts, MAP draws and a fast reproduction of Wagala et al. (2026)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from boolmf import BayesianBooleanMF, BetaMixture, presets
from boolmf._sampler.chain import LevelPrior, _mixture_rates

from .conftest import FAST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "papers"))


def test_mixture_kernel_keeps_its_prior_without_data():
    # with no data the three Gibbs steps must leave the prior joint distribution invariant:
    # start 4000 independent chains (three rates each) from the prior, take five steps, and
    # check P(psi = 1) = E[w] = 2/3 and E[rate] = 2/3 * 5/6 + 1/3 * 1/6
    prior = LevelPrior("mixture", "row", mix=(5.0, 1.0, 1.0, 5.0, 2.0, 1.0))
    rng = np.random.default_rng(0)
    zeros = np.zeros(3, np.int64)
    psis, rates = [], []
    for _ in range(4000):
        state = {"psi": rng.random(3) < rng.beta(2.0, 1.0)}
        for _ in range(5):
            r = _mixture_rates(prior, zeros, zeros, state, rng)
        psis.append(state["psi"].mean())
        rates.append(r.mean())
    assert np.mean(psis) == pytest.approx(2 / 3, abs=0.02)
    assert np.mean(rates) == pytest.approx(2 / 3 * 5 / 6 + 1 / 3 * 1 / 6, abs=0.02)


def test_beta_mixture_validation(small_data):
    X, _ = small_data
    with pytest.raises(ValueError, match="fixed n_components"):
        BayesianBooleanMF(membership_prior=BetaMixture(), membership_level="feature",
                          split_merge=False).fit(X)
    with pytest.raises(ValueError, match="split_merge"):
        BayesianBooleanMF(n_components=3, membership_prior=BetaMixture()).fit(X)
    with pytest.raises(ValueError, match="positive"):
        BayesianBooleanMF(n_components=3, membership_prior=BetaMixture((0, 1)),
                          split_merge=False).fit(X)
    with pytest.raises(ValueError, match="does not take"):
        BayesianBooleanMF(n_components=3, init="asso", init_params={"tau": 0.5},
                          split_merge=False).fit(X)
    with pytest.raises(ValueError, match="store_draws"):
        BayesianBooleanMF(store_draws=0).fit(X)


def test_map_draw_asso_start_and_draw_cap(small_data):
    X, truth = small_data
    K = truth["members"].shape[0]
    m = BayesianBooleanMF(n_components=K, membership_prior=BetaMixture((1, 9), (9, 1)),
                          membership_level="feature", activation_level="sample", init="asso",
                          split_merge=False, store_draws=5, **{**FAST, "n_draws": 40}).fit(X)
    assert m.map_components_.shape == m.components_.shape
    assert m.map_activations_.shape == m.activations_.shape
    assert np.isfinite(m.map_log_posterior_) and 0 < m.map_background_rate_ < 1
    assert set(np.unique(m.map_components_)) <= {0, 1}
    assert all(sum(d["chain"] == c for d in m._draws_) <= 5 for c in range(FAST["n_chains"]))
    # the MAP reconstruction explains the data about as well as the posterior means
    Z = (m.map_activations_.astype(int) @ m.map_components_ > 0)
    observed = ~np.isnan(X)
    assert np.mean(Z[observed] == (X[observed] == 1)) > 0.9
    ibp = BayesianBooleanMF(**FAST).fit(X)
    assert ibp.map_components_ is None


def test_reproduces_wagala2026_scenario1_quickly():
    from wagala2026 import fit_bbmf, load, metrics

    X, truth = load(1)
    m = fit_bbmf(X, sweeps=3000, jobs=1)
    spec, f1, mcc, err = metrics(m.map_activations_.astype(int) @ m.map_components_ > 0, truth)
    # published BBMF: 0.960 / 0.928 / 0.903 / 0.039 (Asso: 0.910 / 0.830 / 0.767 / 0.095)
    assert err <= 0.05 and f1 >= 0.9 and mcc >= 0.87 and spec >= 0.94


def test_wagala_preset_is_read_only():
    with pytest.raises(TypeError):
        presets.wagala2026["n_chains"] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(presets.wagala2026["membership_prior"], BetaMixture)
