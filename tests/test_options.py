"""Option keywords behind the paper presets, and a fast reproduction of Rukat et al. (2017)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from boolmf import BayesianBooleanMF, presets
from boolmf.datasets import make_boolean_factors

from .conftest import FAST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "papers"))


def test_option_validation(small_data):
    X, _ = small_data
    with pytest.raises(ValueError, match="or_flip"):
        BayesianBooleanMF(likelihood="noisy_or", tied_rates=True).fit(X)
    with pytest.raises(ValueError, match="or_flip"):
        BayesianBooleanMF(likelihood="noisy_or", rate_estimation="mle").fit(X)
    with pytest.raises(ValueError, match="split_merge"):
        BayesianBooleanMF(n_components=3, membership_prior=0.2).fit(X)
    with pytest.raises(ValueError, match="fixed n_components"):
        BayesianBooleanMF(activation_prior=0.3).fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(update="annealed").fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(membership_prior="guess", split_merge=False, n_components=3).fit(X)
    with pytest.raises(ValueError, match="'asso' or a .* got 'nndsvd'"):
        BayesianBooleanMF(init="nndsvd").fit(X)


@pytest.mark.parametrize("params", [
    dict(membership_level="shared"),
    dict(membership_level="feature", activation_level="sample"),
    dict(membership_prior=0.1, activation_prior=0.3),
    dict(membership_prior="empirical", activation_prior="empirical", update="metropolised",
         update_order="activations_first", tied_rates=True, rate_estimation="mle",
         init="uniform"),
])
def test_fixed_k_prior_levels_recover_components(small_data, params):
    X, truth = small_data
    K = truth["members"].shape[0]
    m = BayesianBooleanMF(n_components=K, split_merge=False, **{**FAST, **params}).fit(X)
    assert m.components_.shape[1] == X.shape[1]
    assert np.isfinite(m.detection_rate_)
    if params.get("tied_rates"):
        assert np.isclose(m.detection_rate_ + m.background_rate_, 1.0)


def test_fixed_alpha_and_empty_start(small_data):
    X, _ = small_data
    m = BayesianBooleanMF(alpha_prior=2.5, init="empty", **FAST).fit(X)
    assert m.alpha_ == 2.5
    assert m.n_components_ >= 1


@pytest.mark.parametrize("density, flip, published", [(0.5, 0.25, 0.05), (0.7, 0.50, 0.30)])
def test_reproduces_rukat2017_random_factorisation(density, flip, published):
    from rukat2017 import run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, _ = run(100, 7, density, flip, repeats=3)
    assert abs(mean - published) < 0.04


def test_presets_are_read_only():
    with pytest.raises(TypeError):
        presets.rukat2017["n_chains"] = 4
    X = make_boolean_factors(n_samples=60, n_features=50, n_components=3, random_state=0)
    params = {**presets.rukat2017, "n_draws": 20}
    m = BayesianBooleanMF(n_components=3, random_state=0, **params).fit(X)
    assert m.chain_status_.shape == (1,)
