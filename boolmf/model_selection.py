"""Entry-wise (matrix-completion) cross-validation.

scikit-learn's splitters hold out whole samples, which tests projection of new samples.
These splitters hold out individual entries instead: the model is fit with those entries
masked and scored on them.
"""

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state

__all__ = ["EntryKFold", "EntryShuffleSplit", "cross_validate_entries"]


def _observed(X):
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype == bool:
        return np.ones(X.shape, bool), X
    X = X.astype(float)
    obs = ~np.isnan(X)
    return obs, np.where(obs, X, 0) > 0


def _repair(test, obs):
    """Unmask one entry in any row or column whose observed entries are all held out."""
    rng = np.random.default_rng(0)
    for axis in (0, 1):
        o = obs.sum(axis=axis)
        t = (test & obs).sum(axis=axis)
        for idx in np.flatnonzero((o > 0) & (t == o)):
            if axis == 1:
                cols = np.flatnonzero(test[idx] & obs[idx])
                test[idx, rng.choice(cols)] = False
            else:
                rows = np.flatnonzero(test[:, idx] & obs[:, idx])
                test[rng.choice(rows), idx] = False
    return test


class EntryShuffleSplit:
    """Random held-out entries, stratified by presence.

    Parameters
    ----------
    n_splits : int, default=1
    test_size : float, default=0.2
    random_state : int, RandomState instance or None
    """

    def __init__(self, n_splits=1, *, test_size=0.2, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X):
        """Yield boolean masks of held-out entries (True = held out)."""
        rng = check_random_state(self.random_state)
        obs, pos = _observed(X)
        for _ in range(self.n_splits):
            test = np.zeros(obs.shape, bool)
            for cls in (pos & obs, ~pos & obs):
                idx = np.flatnonzero(cls)
                k = int(round(self.test_size * idx.size))
                test.flat[rng.choice(idx, size=k, replace=False)] = True
            yield _repair(test, obs)

    def get_n_splits(self, X=None):
        return self.n_splits


class EntryKFold:
    """Partition observed entries into ``n_splits`` folds, stratified by presence.

    Parameters
    ----------
    n_splits : int, default=5
    shuffle : bool, default=True
    random_state : int, RandomState instance or None
    """

    def __init__(self, n_splits=5, *, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """Yield boolean masks of held-out entries (True = held out)."""
        rng = check_random_state(self.random_state)
        obs, pos = _observed(X)
        fold = np.full(obs.shape, -1)
        for cls in (pos & obs, ~pos & obs):
            idx = np.flatnonzero(cls)
            if self.shuffle:
                idx = rng.permutation(idx)
            fold.flat[idx] = np.arange(idx.size) % self.n_splits
        for f in range(self.n_splits):
            yield _repair(fold == f, obs)

    def get_n_splits(self, X=None):
        return self.n_splits


def cross_validate_entries(estimator, X, cv=None, threshold=None):
    """Fit on entries not held out and score the held-out ones, for each split.

    Parameters
    ----------
    estimator : BoolMF
    X : array-like of shape (n_samples, n_features)
    cv : EntryKFold, EntryShuffleSplit, int or None
        An int means ``EntryKFold(n_splits=cv)``; None means 5 folds.
    threshold : float or None
        Passed to :func:`boolmf.metrics.confusion_table`.

    Returns
    -------
    dict of arrays, one value per split: ``log_likelihood`` (mean held-out log predictive
    density) and the held-out confusion rates.
    """
    from .metrics import confusion_table

    if cv is None:
        cv = EntryKFold(5)
    elif isinstance(cv, int):
        cv = EntryKFold(cv)
    out = {k: [] for k in ("log_likelihood", "precision", "recall", "specificity", "npv")}
    for test in cv.split(X):
        est = clone(estimator).fit(X, mask=test)
        out["log_likelihood"].append(est.score(X, entries=test))
        tab = confusion_table(est, X, entries=test, threshold=threshold)
        for k in ("precision", "recall", "specificity", "npv"):
            out[k].append(float(tab[k]))
    return {k: np.asarray(v) for k, v in out.items()}

