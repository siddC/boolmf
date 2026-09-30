"""Standard (non-Bayesian) Boolean matrix factorization."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._algorithms import asso, asso_usage, grecond, grecond_usage
from .utils.validation import to_binary_int8

ALGORITHMS = ("asso", "grecond")


class BooleanMF(TransformerMixin, BaseEstimator):
    """Boolean matrix factorization.

    Finds binary matrices W (samples x components) and H (components x features) whose
    Boolean product, (W H)_ij = OR_k (W_ik AND H_kj), approximates a binary matrix X: each
    component is a set of features (a row of H, stored in ``components_``) and each sample uses
    a set of components (a row of W, returned by ``transform``). The algorithms are
    deterministic combinatorial searches; for posterior probabilities, noise rates and a
    learned number of components see ``BayesianBooleanMF``.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components. None lets the algorithm stop by itself: ``"grecond"`` at an
        exact cover (or at ``coverage``), ``"asso"`` when no candidate improves its cover
        function.
    algorithm : {"asso", "grecond"}, default="asso"
        ``"asso"``: Miettinen et al. (2008), The Discrete Basis Problem. Candidate components
        come from pairwise feature associations (confidence at least ``threshold``) and are
        chosen greedily to maximize ``positive_weight`` x (ones covered) - ``negative_weight``
        x (zeros covered); each sample uses a component when that raises its own score.
        ``"grecond"``: Belohlavek & Vychodil (2010), greedy formal concepts. Components never
        cover a zero ("from below") and, run to the end, reproduce X exactly.
    threshold : float in (0, 1], default=0.5
        ``"asso"`` only: the association threshold tau. The paper tunes it per dataset.
    positive_weight, negative_weight : float, default=1.0
        ``"asso"`` only: the weights w+ and w- of covered ones and covered zeros.
    coverage : float in (0, 1], default=1.0
        Stop once this fraction of the ones in X is covered.
    binarize : float or None, default=None
        None requires 0/1 input. A float maps values above it to 1.

    Attributes
    ----------
    components_ : ndarray of shape (n_components_, n_features), dtype uint8
        The components H: which features each component contains.
    n_components_ : int
        Number of components found.
    reconstruction_err_ : int
        Number of entries where the Boolean product of the training usage and the components
        differs from X (Hamming distance).
    coverage_ : float
        Fraction of the ones in X covered by the training reconstruction.
    n_features_in_ : int
    feature_names_in_ : ndarray of str
        Only when X has string feature names.

    Notes
    -----
    Missing entries are not supported. ``fit_transform`` returns the usage found while fitting,
    which equals ``transform`` of the training data. For new data, ``transform`` applies the
    algorithm's own usage rule to the fitted components: ``"asso"`` adds them in order, each
    used by a sample when it raises the sample's cover score; ``"grecond"`` uses a component in
    every sample that has all of its features.
    """

    def __init__(self, n_components=None, *, algorithm="asso", threshold=0.5,
                 positive_weight=1.0, negative_weight=1.0, coverage=1.0, binarize=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.threshold = threshold
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.coverage = coverage
        self.binarize = binarize

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = []        # usage is 0/1 uint8 whatever X is
        return tags

    def _check_params(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}; got {self.algorithm!r}.")
        if self.n_components is not None and (
                not isinstance(self.n_components, numbers.Integral) or self.n_components < 1):
            raise ValueError(f"n_components must be a positive int or None; got "
                             f"{self.n_components!r}.")
        if not (isinstance(self.threshold, numbers.Real) and 0 < self.threshold <= 1):
            raise ValueError(f"threshold must be in (0, 1]; got {self.threshold!r}.")
        for name in ("positive_weight", "negative_weight"):
            value = getattr(self, name)
            if not (isinstance(value, numbers.Real) and value > 0):
                raise ValueError(f"{name} must be a positive number; got {value!r}.")
        if not (isinstance(self.coverage, numbers.Real) and 0 < self.coverage <= 1):
            raise ValueError(f"coverage must be in (0, 1]; got {self.coverage!r}.")
        if self.binarize is not None and not isinstance(self.binarize, numbers.Real):
            raise ValueError("binarize must be a number or None.")

    def _validate_X(self, X, reset):
        X = validate_data(self, X, accept_sparse=("csr", "csc"), dtype="numeric", reset=reset)
        V = to_binary_int8(X, binarize=self.binarize)
        return V == 1

    def fit(self, X, y=None):
        """Fit the factorization.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Binary data (or any numbers with ``binarize`` set).
        y : ignored

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the factorization and return the usage W of the training samples.

        Returns
        -------
        ndarray of shape (n_samples, n_components_), dtype uint8
        """
        self._check_params()
        X = self._validate_X(X, reset=True)
        if self.algorithm == "asso":
            W, H = asso(X, self.n_components, float(self.threshold),
                        float(self.positive_weight), float(self.negative_weight),
                        float(self.coverage))
        else:
            W, H = grecond(X, self.n_components, float(self.coverage))
        self.components_ = H.astype(np.uint8)
        self.n_components_ = H.shape[0]
        R = (W.astype(np.int64) @ H.astype(np.int64)) > 0
        self.reconstruction_err_ = int(np.count_nonzero(R != X))
        self.coverage_ = float(np.count_nonzero(R & X) / max(1, np.count_nonzero(X)))
        return W.astype(np.uint8)

    def transform(self, X):
        """Usage W of the fitted components by (new) samples.

        Returns
        -------
        ndarray of shape (n_samples, n_components_), dtype uint8
        """
        check_is_fitted(self, "components_")
        X = self._validate_X(X, reset=False)
        H = self.components_.astype(bool)
        if self.algorithm == "asso":
            W = asso_usage(X, H, float(self.positive_weight), float(self.negative_weight))
        else:
            W = grecond_usage(X, H)
        return W.astype(np.uint8)

    def inverse_transform(self, X):
        """Boolean product of a usage matrix and the components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components_)
            0/1 usage, for example from ``transform``.

        Returns
        -------
        ndarray of shape (n_samples, n_features), dtype uint8
        """
        check_is_fitted(self, "components_")
        W = np.asarray(X)
        if W.ndim != 2 or W.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"expected shape (n_samples, {self.components_.shape[0]}); got {W.shape}.")
        return ((W != 0).astype(np.int64) @ self.components_.astype(np.int64) > 0).astype(
            np.uint8)

    def get_feature_names_out(self, input_features=None):
        """Output names for ``transform``: ``booleanmf0``, ``booleanmf1``, ..."""
        check_is_fitted(self, "components_")
        return np.asarray([f"booleanmf{i}" for i in range(self.n_components_)], dtype=object)
