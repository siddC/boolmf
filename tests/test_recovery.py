"""Recovery on a moderate simulation (about 30 s on two cores). Run with ``pytest -m slow``."""

import warnings

import numpy as np
import pytest

from boolmf import BoolMF
from boolmf.datasets import make_boolean_factors
from boolmf.matching import jaccard_matrix
from boolmf.metrics import confusion_table


@pytest.mark.slow
def test_recovery_moderate_simulation():
    X, truth = make_boolean_factors(n_samples=300, n_features=400, n_components=6, missing=0.05,
                                    random_state=0, return_truth=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no convergence warnings expected
        model = BoolMF(n_chains=4, n_jobs=-1, max_sweeps=3000, random_state=0).fit(X)
    assert model.n_components_ == 6
    members, active = model.binarize_components()
    robust = model.component_flags_ == "robust"
    assert jaccard_matrix(truth["members"], members[robust]).max(axis=1).min() > 0.95
    assert jaccard_matrix(truth["activations"].T, active[:, robust].T).max(axis=1).min() > 0.95
    assert abs(model.detection_rate_ - 0.97) < 0.02
    table = confusion_table(model)
    assert table["precision"] > 0.95 and table["recall"] > 0.95
    assert np.all(model.chain_status_ != "stuck")
