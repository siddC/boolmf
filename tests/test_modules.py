import numpy as np
import pytest

from boolmf import BoolMF
from boolmf.datasets import make_boolean_factors, make_nested_factors
from boolmf.diagnostics import ess, geweke, rhat
from boolmf.matching import jaccard_matrix, match_components
from boolmf.metrics import calibration_curve, component_integrity, confusion_table
from boolmf.model_selection import EntryKFold, EntryShuffleSplit, cross_validate_entries

from .conftest import FAST


def test_rhat_and_ess():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 1000))
    assert rhat(iid) == pytest.approx(1.0, abs=0.01)
    shifted = iid + np.arange(4)[:, None]
    assert rhat(shifted) > 1.5
    assert 3000 < ess(iid) < 5000
    ar = np.zeros(4000)
    for t in range(1, 4000):
        ar[t] = 0.95 * ar[t - 1] + rng.normal()
    assert ess(ar[None]) < 400
    assert geweke(iid[0]) < 3
    assert geweke(np.linspace(0, 10, 1000) + rng.normal(size=1000)) > 3


def test_matching():
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], bool)
    B = np.array([[0, 0, 1, 1], [1, 1, 1, 0]], bool)
    J = jaccard_matrix(A, B)
    assert J[1, 0] == 1.0 and J[0, 1] == pytest.approx(2 / 3)
    pairs = match_components(A, B)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert match_components(A, np.zeros((0, 4), bool)) == []


def test_entry_splitters(small_data):
    X, _ = small_data
    folds = list(EntryKFold(5, random_state=0).split(X))
    assert len(folds) == 5
    total = np.sum(folds, axis=0)
    assert total.max() <= 1 and total.sum() >= 0.99 * X.size
    for f in folds:
        assert np.all(f.sum(1) < X.shape[1]) and np.all(f.sum(0) < X.shape[0])
    (test,) = list(EntryShuffleSplit(test_size=0.2, random_state=0).split(X))
    assert 0.18 < test.mean() < 0.22
    assert abs(X[test].mean() - X.mean()) < 0.02


def test_cross_validate_entries(small_data):
    X, _ = small_data
    res = cross_validate_entries(BoolMF(**FAST), X, cv=EntryShuffleSplit(2, test_size=0.1,
                                                                          random_state=0))
    assert set(res) == {"log_likelihood", "precision", "recall", "specificity", "npv"}
    assert res["log_likelihood"].shape == (2,)
    assert np.all(res["recall"] > 0.8)


def test_metrics(small_data, fitted):
    X, _ = small_data
    tab = confusion_table(fitted)
    n_obs = X.size
    assert tab["TP"] + tab["FP"] + tab["FN"] + tab["TN"] == pytest.approx(n_obs)
    assert tab["recall"] > 0.9 and tab["specificity"] > 0.95
    per_sample = confusion_table(fitted, level="sample", threshold=0.5)
    assert per_sample["TP"].shape == (X.shape[0],)
    integ = component_integrity(fitted)
    robust = fitted.component_flags_ == "robust"
    assert np.all(integ[robust] > 0.9)
    pred, obs, counts = calibration_curve(fitted, n_bins=5)
    assert counts.sum() == n_obs
    assert np.nanmax(np.abs(pred - obs)[counts > 100]) < 0.1


def test_datasets():
    X, t = make_boolean_factors(50, 40, 3, missing=0.1, random_state=0, return_truth=True)
    assert X.shape == (50, 40) and np.isnan(X).any()
    assert t["members"].shape == (3, 40) and t["activations"].shape == (50, 3)
    X, t = make_nested_factors(random_state=0, return_truth=True)
    assert X.shape[0] == 500 and t["members"].shape == (20, X.shape[1])
    assert len(t["names"]) == 20
