import numpy as np
import pytest
from scipy import stats

from boolmf import AnchorComponent, BayesianBooleanMF
from boolmf.datasets import make_boolean_factors
from boolmf.matching import jaccard_matrix

from .conftest import FAST


def _best_jaccard(truth_members, members):
    return jaccard_matrix(truth_members, members).max(axis=1)


def test_recovers_components(small_data, fitted):
    X, truth = small_data
    members, _ = fitted.binarize_components()
    robust = fitted.component_flags_ == "robust"
    assert fitted.n_components_ == truth["members"].shape[0]
    assert np.all(_best_jaccard(truth["members"], members[robust]) >= 0.9)
    assert 0.9 < fitted.detection_rate_ < 1.0
    assert 0.0 < fitted.background_rate_ < 0.05


def test_fitted_attribute_shapes(small_data, fitted):
    X, _ = small_data
    K = fitted.components_.shape[0]
    assert fitted.components_.shape == (K, X.shape[1])
    assert fitted.activations_.shape == (X.shape[0], K)
    assert fitted.robustness_.shape == fitted.prevalence_.shape == (K,)
    assert set(fitted.component_flags_) <= {"anchor", "robust", "low_support", "not_robust"}
    assert fitted.log_likelihood_trace_.shape[0] == 2
    assert set(fitted.rhat_) == set(fitted.ess_)
    assert set(fitted.split_merge_acceptance_) == {
        "split", "merge", "reallocate", "factor", "unfactor"}
    assert fitted.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("likelihood", ["noisy_or", "or_flip"])
def test_both_likelihoods_run(small_data, likelihood):
    X, truth = small_data
    m = BayesianBooleanMF(likelihood=likelihood, **FAST).fit(X)
    members, _ = m.binarize_components()
    robust = m.component_flags_ == "robust"
    assert np.all(_best_jaccard(truth["members"], members[robust]) >= 0.8)


def test_fixed_number_of_components(small_data):
    X, _ = small_data
    m = BayesianBooleanMF(n_components=3, **FAST).fit(X)
    assert m.alpha_ is None
    assert m.components_.shape[0] <= 3


def test_determinism_and_jobs(small_data):
    X, _ = small_data
    a = BayesianBooleanMF(**FAST).fit(X)
    b = BayesianBooleanMF(**{**FAST, "n_jobs": 2}).fit(X)
    np.testing.assert_array_equal(a.components_, b.components_)
    np.testing.assert_array_equal(a.activations_, b.activations_)


def test_masked_entries_do_not_affect_fit(small_data):
    X, _ = small_data
    rng = np.random.default_rng(1)
    mask = rng.random(X.shape) < 0.1
    X2 = X.copy()
    X2[mask] = 1 - X2[mask]
    a = BayesianBooleanMF(**FAST).fit(X, mask=mask)
    b = BayesianBooleanMF(**FAST).fit(X2, mask=mask)
    np.testing.assert_array_equal(a.components_, b.components_)
    X3 = X.copy()
    X3[mask] = np.nan
    c = BayesianBooleanMF(**FAST).fit(X3)
    np.testing.assert_array_equal(a.components_, c.components_)


def test_transform_is_row_order_invariant(small_data, fitted):
    X, _ = small_data
    perm = np.random.default_rng(0).permutation(X.shape[0])
    Z = fitted.transform(X)
    Zp = fitted.transform(X[perm])
    np.testing.assert_allclose(Zp, Z[perm])
    np.testing.assert_allclose(fitted.transform(X[:7]), Z[:7])


def test_transform_close_to_training_activations(small_data, fitted):
    X, _ = small_data
    robust = fitted.component_flags_ == "robust"
    Z = fitted.transform(X)
    assert np.mean(np.abs(Z[:, robust] - fitted.activations_[:, robust])) < 0.05


def test_inverse_transform_and_scores(small_data, fitted):
    X, _ = small_data
    P = fitted.inverse_transform(fitted.activations_)
    assert P.shape == X.shape
    assert np.all((P > 0) & (P < 1))
    s = fitted.score(X)
    assert np.isfinite(s) and s < 0
    ss = fitted.score_samples(X)
    assert ss.shape == (X.shape[0],)
    E = fitted.explained_probability()
    assert E.shape == X.shape and E.min() >= 0 and E.max() <= 1


def test_score_on_held_out_entries(small_data):
    X, _ = small_data
    mask = np.random.default_rng(2).random(X.shape) < 0.1
    m = BayesianBooleanMF(**FAST).fit(X, mask=mask)
    held = m.score(X, entries=mask)
    base_rate = X[~mask].mean()
    naive = np.mean(np.where(X[mask] == 1, np.log(base_rate), np.log1p(-base_rate)))
    assert held > naive + 0.1


def test_binarize_components_methods(fitted):
    m1, a1 = fitted.binarize_components()
    m2, a2 = fitted.binarize_components(method="bfdr", fdr=0.01)
    assert m1.dtype == bool and a1.shape == fitted.activations_.shape
    assert m2.shape == m1.shape
    with pytest.raises(ValueError):
        fitted.binarize_components(method="nope")


def test_get_draws_and_feature_names(fitted):
    draws = list(fitted.get_draws("components"))
    assert len(draws) == 2 * 40
    assert draws[0].shape == fitted.components_.shape
    acts = next(fitted.get_draws("activations"))
    assert acts.shape == fitted.activations_.shape
    names = fitted.get_feature_names_out()
    assert names[0] == "bayesianbooleanmf0" and len(names) == fitted.components_.shape[0]


def test_summary(fitted):
    s = fitted.summary()
    assert {"flag", "robustness", "prevalence", "integrity", "leakage"} <= set(s.columns)


def test_anchor_component(small_data):
    X, _ = small_data
    Xa = np.hstack([np.ones((X.shape[0], 5)), X])
    Xa[3, 0] = 0
    m = BayesianBooleanMF(anchor_components=[AnchorComponent(members=np.arange(5))], **FAST).fit(Xa)
    assert m.component_flags_[0] == "anchor"
    np.testing.assert_allclose(m.activations_[:, 0], 1.0)
    np.testing.assert_allclose(m.components_[0, :5], 1.0)
    np.testing.assert_allclose(m.components_[0, 5:], 0.0)


def test_inits(small_data):
    X, truth = small_data
    m = BayesianBooleanMF(init="nmf", init_params={"n_components": 3}, **FAST).fit(X)
    assert m.n_components_ >= 1
    init = (truth["members"], truth["activations"])
    m2 = BayesianBooleanMF(init=init, **FAST).fit(X)
    members, _ = m2.binarize_components()
    robust = m2.component_flags_ == "robust"
    assert np.all(_best_jaccard(truth["members"], members[robust]) >= 0.9)


def test_rate_priors(small_data):
    X, _ = small_data
    background = stats.truncnorm(-2, 2, loc=0.02, scale=0.01)
    m = BayesianBooleanMF(detection_prior=0.95, background_prior=background,
                          likelihood="noisy_or", **FAST).fit(X)
    assert m.detection_rate_ == pytest.approx(0.95)
    with pytest.raises(ValueError):
        BayesianBooleanMF(detection_prior=stats.norm(0, 1), **FAST).fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(detection_prior=1.5, **FAST).fit(X)


def test_input_validation(small_data):
    X, _ = small_data
    with pytest.raises(ValueError, match="binary"):
        BayesianBooleanMF(**FAST).fit(X * 2)
    BayesianBooleanMF(binarize=0.5, **FAST).fit(X * 2)
    with pytest.raises(ValueError):
        BayesianBooleanMF(likelihood="gaussian").fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(n_chains=0).fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(split_merge=-1).fit(X)


def test_sparse_and_dataframe_input(small_data):
    import pandas as pd
    import scipy.sparse as sp

    X, _ = small_data
    a = BayesianBooleanMF(**FAST).fit(sp.csr_matrix(X))
    b = BayesianBooleanMF(**FAST).fit(X)
    np.testing.assert_array_equal(a.components_, b.components_)
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(X.shape[1])])
    c = BayesianBooleanMF(**FAST).fit(df)
    assert list(c.feature_names_in_[:2]) == ["g0", "g1"]


def test_split_merge_can_be_turned_off(small_data):
    X, truth = small_data
    m = BayesianBooleanMF(split_merge=False, **FAST).fit(X)
    assert all(np.isnan(v) for v in m.split_merge_acceptance_.values())
    members, _ = m.binarize_components()
    robust = m.component_flags_ == "robust"
    assert jaccard_matrix(truth["members"], members[robust]).max(axis=1).min() > 0.9


def test_per_sample_rates():
    rng = np.random.default_rng(0)
    n = 150
    det = 1 / (1 + np.exp(-(np.log(9.0) + rng.normal(0, 1.0, n))))
    bg = 1 / (1 + np.exp(-(np.log(0.01 / 0.99) + rng.normal(0, 0.7, n))))
    X = make_boolean_factors(n_samples=n, n_features=200, n_components=4, detection=det,
                             background=bg, random_state=1)
    m = BayesianBooleanMF(detection_effects=("sample",), background_effects="sample", **FAST).fit(X)
    assert m.detection_rate_per_sample_.shape == m.background_rate_per_sample_.shape == (n,)
    lo, hi = m.detection_rate_per_sample_interval_.T
    assert np.all(lo <= m.detection_rate_per_sample_) and np.all(m.detection_rate_per_sample_ <= hi)
    assert np.all(m.background_rate_per_sample_ < m.detection_rate_per_sample_)   # or_flip order
    assert np.corrcoef(m.detection_rate_per_sample_, det)[0, 1] > 0.5
    assert 0.3 < m.detection_spread_ < 3.0
    # refitting without effects removes the per-sample attributes
    m.set_params(detection_effects=(), background_effects=()).fit(X)
    assert not hasattr(m, "detection_rate_per_sample_")


def test_rate_effect_validation(small_data):
    X, _ = small_data
    with pytest.raises(NotImplementedError, match="v0.3"):
        BayesianBooleanMF(detection_effects=("feature",)).fit(X)
    with pytest.raises(ValueError, match="requires likelihood='noisy_or'"):
        BayesianBooleanMF(detection_effects=("component",)).fit(X)        # default or_flip
    with pytest.raises(ValueError):
        BayesianBooleanMF(background_effects=("component",)).fit(X)
    with pytest.raises(ValueError, match="fixes"):
        BayesianBooleanMF(detection_effects=("sample",), detection_prior=0.9).fit(X)


def test_per_component_rates_noisy_or():
    lam = np.array([0.6, 0.8, 0.97])
    X, truth = make_boolean_factors(n_samples=150, n_features=200, n_components=3,
                                    component_detection=lam, background=0.01, random_state=2,
                                    return_truth=True)
    m = BayesianBooleanMF(likelihood="noisy_or", detection_effects=("component",), **FAST).fit(X)
    K = m.components_.shape[0]
    assert m.detection_rate_per_component_.shape == (K,)
    assert m.detection_rate_per_component_interval_.shape == (K, 2)
    robust = np.flatnonzero(m.component_flags_ == "robust")
    members, _ = m.binarize_components()
    J = jaccard_matrix(truth["members"], members[robust])
    est = m.detection_rate_per_component_[robust[J.argmax(1)]]
    assert np.all(np.abs(est - lam) < 0.1)
    assert m.detection_component_spread_ > 0
    assert m.transform(X[:5]).shape == (5, K)                    # projection path
    assert np.all(np.isfinite(m.inverse_transform(m.transform(X[:5]))))
    m2 = BayesianBooleanMF(likelihood="noisy_or", detection_effects=("sample", "component"),
                           background_effects=("sample",), **FAST).fit(X)
    assert m2.detection_rate_per_sample_.shape == (150,)
    assert m2.background_rate_per_sample_.shape == (150,)
    assert np.isfinite(m2.detection_rate_per_component_[m2.component_flags_ == "robust"]).all()


def test_redundant_components_are_detected():
    from boolmf._bayesian import _redundant_components

    A = np.zeros((40, 4))
    A[:20, 0] = A[:20, 1] = 1              # 0 and 1: same carriers, different members
    A[20:, 2] = 1
    A[20:30, 3] = 1                        # 3 lies inside 2 ...
    B = np.zeros((4, 30))
    B[0, :10] = B[1, 10:20] = 1
    B[2, 20:] = 1
    B[3, 22:28] = 1                        # ... on both sides, so 2 covers all it covers
    flags = np.array(["robust"] * 4)
    found = {(kind, k, l) for kind, k, l, _ in _redundant_components(B, A, flags)}
    assert found == {("carriers", 0, 1), ("covered", 3, -1)}
    flags[3] = "low_support"               # only robust components are checked
    found = {(kind, k, l) for kind, k, l, _ in _redundant_components(B, A, flags)}
    assert found == {("carriers", 0, 1)}
    A[:10, 1] = 0                          # carriers of 0 and 1 now differ enough
    assert _redundant_components(B, A, flags) == []
