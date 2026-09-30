"""BooleanMF: Asso and GreConD, checked against published results on bundled UCI data."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from boolmf import BooleanMF
from boolmf.datasets import make_boolean_factors

DATA = Path(__file__).resolve().parent / "data"


def _load(name):
    """UCI Mushroom (8124 x 119) or Tic-tac-toe (958 x 29), one-hot with the class, from PMLB."""
    d = np.load(DATA / f"{name}.npz")
    return np.unpackbits(d["bits"], axis=1, count=int(d["shape"][1])).astype(bool)


def _factors_needed(X, W, H, levels):
    covered = np.zeros(X.shape, bool)
    curve = []
    for k in range(H.shape[0]):
        covered |= np.outer(W[:, k], H[k])
        curve.append(1.0 - np.count_nonzero(covered != X) / X.sum())
    curve = np.array(curve)
    return [int(np.argmax(curve >= c - 1e-12)) + 1 if (curve >= c - 1e-12).any() else None
            for c in levels]


def test_grecond_reproduces_tic_tac_toe_counts():
    # Belohlavek & Trnecka (2015), Table 4
    X = _load("tic_tac_toe")
    m = BooleanMF(algorithm="grecond")
    W = m.fit_transform(X).astype(bool)
    assert _factors_needed(X, W, m.components_.astype(bool),
                           (0.25, 0.5, 0.75, 0.95, 1.0)) == [5, 12, 19, 28, 32]
    assert m.reconstruction_err_ == 0 and m.coverage_ == 1.0


def test_grecond_reproduces_mushroom_counts():
    X = _load("mushroom")
    m = BooleanMF(algorithm="grecond")
    W = m.fit_transform(X).astype(bool)
    got = _factors_needed(X, W, m.components_.astype(bool), (0.25, 0.5, 0.75, 0.95, 1.0))
    assert got[:3] == [3, 7, 24]
    assert abs(got[3] - 63) <= 1 and abs(got[4] - 120) <= 1


def test_asso_reproduces_mushroom_counts():
    # Belohlavek & Trnecka (2015), Table 4: 2 / 6 / 36 factors, never 95%
    X = _load("mushroom")
    m = BooleanMF(60, algorithm="asso", threshold=0.95)
    W = m.fit_transform(X).astype(bool)
    got = _factors_needed(X, W, m.components_.astype(bool), (0.25, 0.5, 0.75, 0.95))
    assert got == [2, 6, 36, None]


@pytest.mark.parametrize("algorithm", ["asso", "grecond"])
def test_transform_matches_fit_and_recovers_planted_factors(algorithm):
    rng = np.random.default_rng(0)
    members = np.zeros((4, 60), bool)
    for k in range(4):
        members[k, 12 * k:12 * k + 12] = True           # four blocks, 12 unused features
    X = ((rng.random((200, 4)) < 0.3).astype(int) @ members.astype(int) > 0).astype(float)
    m = BooleanMF(4 if algorithm == "asso" else None, algorithm=algorithm, threshold=0.9)
    W = m.fit_transform(X)
    np.testing.assert_array_equal(W, m.transform(X))
    np.testing.assert_array_equal(m.inverse_transform(W) == 1, (W @ m.components_) > 0)
    assert m.reconstruction_err_ == int(np.count_nonzero(m.inverse_transform(W) != X))
    if algorithm == "grecond":
        assert m.reconstruction_err_ == 0
    found = {tuple(np.flatnonzero(h)) for h in m.components_}
    planted = {tuple(np.flatnonzero(u)) for u in members}
    assert planted <= found


def test_stopping_rules():
    X = _load("tic_tac_toe")
    assert BooleanMF(7, algorithm="grecond").fit(X).n_components_ == 7
    m = BooleanMF(algorithm="grecond", coverage=0.5).fit(X)
    assert m.n_components_ == 12 and m.coverage_ >= 0.5
    m = BooleanMF(algorithm="asso", threshold=0.95)            # stops by itself
    W = m.fit_transform(X)
    assert m.n_components_ >= 1 and W.any(axis=0).all()        # every component is used
    assert BooleanMF(40, algorithm="asso").fit(X).n_components_ == 40   # unused ones kept


def test_input_handling():
    X = make_boolean_factors(n_samples=50, n_features=30, n_components=3, random_state=1)
    dense = BooleanMF(3).fit(X)
    sparse = BooleanMF(3).fit(sp.csr_matrix(X))
    np.testing.assert_array_equal(dense.components_, sparse.components_)
    scores = X * 0.8 + 0.1
    np.testing.assert_array_equal(BooleanMF(3, binarize=0.5).fit(scores).components_,
                                  dense.components_)
    Xn = X.astype(float)
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        BooleanMF(3).fit(Xn)
    with pytest.raises(ValueError, match="binary"):
        BooleanMF(3).fit(scores)
    for params in (dict(algorithm="panda"), dict(n_components=0), dict(threshold=0.0),
                   dict(threshold=1.5), dict(positive_weight=0), dict(coverage=0)):
        with pytest.raises(ValueError):
            BooleanMF(**params).fit(X)
    names = dense.get_feature_names_out()
    assert names[0] == "booleanmf0" and len(names) == dense.n_components_
    with pytest.raises(ValueError, match="shape"):
        dense.inverse_transform(np.ones((2, 5)))


def test_mebf_recovers_blocks_and_stops_by_itself():
    rng = np.random.default_rng(1)
    members = np.zeros((3, 45), bool)
    for k in range(3):
        members[k, 15 * k:15 * k + 15] = True
    usage = rng.random((150, 3)) < 0.3
    X = (usage.astype(int) @ members.astype(int)) > 0
    m = BooleanMF(algorithm="mebf", threshold=0.6)
    W = m.fit_transform(X)
    assert m.reconstruction_err_ == 0
    assert {tuple(np.flatnonzero(h)) for h in m.components_} == \
        {tuple(np.flatnonzero(u)) for u in members}
    np.testing.assert_array_equal(W.astype(bool), usage[:, [
        next(k for k in range(3) if (members[k] == h).all()) for h in m.components_.astype(bool)
    ]])
    assert BooleanMF(2, algorithm="mebf").fit(X).n_components_ == 2


def test_mebf_follows_paper_on_low_density_simulation():
    # wan2020.py: the paper's algorithm without noise, 100 x 100, five patterns, p0 = 0.2
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "papers"))
    from wan2020 import errors

    e = errors(100, 0.2, 0.0, runs=10)
    assert np.all(np.diff(e) < 0) and e[-1] < 0.1
