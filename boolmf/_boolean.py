"""Standard (non-Bayesian) Boolean matrix factorization."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._algorithms import asso, asso_usage, grecond, grecond_usage, panda
from .utils.validation import to_binary_int8

ALGORITHMS = ("asso", "grecond", "panda")
COSTS = ("je", "jp", "ja")
ITEM_ORDERS = ("correlation", "frequency", "couples")


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
        function, ``"panda"`` when a new component would raise its cost.
    algorithm : {"asso", "grecond", "panda"}, default="asso"
        ``"asso"``: Miettinen et al. (2008), The Discrete Basis Problem. Candidate components
        come from pairwise feature associations (confidence at least ``threshold``) and are
        chosen greedily to maximize ``positive_weight`` x (ones covered) - ``negative_weight``
        x (zeros covered); each sample uses a component when that raises its own score.
        ``"grecond"``: Belohlavek & Vychodil (2010), greedy formal concepts. Components never
        cover a zero ("from below") and, run to the end, reproduce X exactly.
        ``"panda"``: PANDA+ (Lucchese et al. 2014). Each component starts as a noise-free core
        of items taken in ``item_order`` and is then extended with rows and items while the
        cost function ``cost`` does not rise and the noise stays within ``row_tolerance`` and
        ``column_tolerance``. With the MDL cost (``"je"``) it also chooses the number of
        components.
    threshold : float in (0, 1], default=0.5
        ``"asso"`` only: the association threshold tau. The paper tunes it per dataset.
    positive_weight, negative_weight : float, default=1.0
        ``"asso"`` only: the weights w+ and w- of covered ones and covered zeros.
    coverage : float in (0, 1], default=1.0
        Stop once this fraction of the ones in X is covered.
    cost : {"je", "jp", "ja"}, default="je"
        ``"panda"`` only: ``"je"`` the MDL (Typed XOR) encoding length of the components and
        the errors (Miettinen & Vreeken 2011); ``"jp"`` ``rho`` x (total size of the
        components' row and column sets) + errors; ``"ja"`` the number of errors.
    rho : float, default=1.0
        ``"panda"`` with ``cost="jp"``: weight of the components' size.
    row_tolerance, column_tolerance : float in [0, 1], default=1.0
        ``"panda"`` only: the noise thresholds eps_r and eps_c. Every row of a component must
        hold at least (1 - eps_r) of its items and every item at least (1 - eps_c) of its rows;
        1 turns the constraint off.
    item_order : {"correlation", "frequency", "couples"}, default="correlation"
        ``"panda"`` only: the order in which items are tried when a core is built.
    n_rounds : int, default=0
        ``"panda"`` only: randomized rounds per component (0: one deterministic round).
    random_state : int, RandomState instance or None, default=None
        ``"panda"`` with ``n_rounds`` > 0: seed of the randomized item orders.
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
    which for Asso and GreConD equals ``transform`` of the training data. For new data,
    ``transform`` applies the algorithm's own usage rule to the fitted components: ``"asso"``
    adds them in order, each used by a sample when it raises the sample's cover score;
    ``"grecond"`` uses a component in every sample that has all of its features. PANDA+ has
    no rule for new samples, so ``transform`` uses Asso's with equal weights; for
    the training data it can differ from the usage found while fitting.
    """

    def __init__(self, n_components=None, *, algorithm="asso", threshold=0.5,
                 positive_weight=1.0, negative_weight=1.0, coverage=1.0, cost="je", rho=1.0,
                 row_tolerance=1.0, column_tolerance=1.0, item_order="correlation",
                 n_rounds=0, random_state=None, binarize=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.threshold = threshold
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.coverage = coverage
        self.cost = cost
        self.rho = rho
        self.row_tolerance = row_tolerance
        self.column_tolerance = column_tolerance
        self.item_order = item_order
        self.n_rounds = n_rounds
        self.random_state = random_state
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
        if self.cost not in COSTS:
            raise ValueError(f"cost must be one of {COSTS}; got {self.cost!r}.")
        if self.item_order not in ITEM_ORDERS:
            raise ValueError(f"item_order must be one of {ITEM_ORDERS}; got "
                             f"{self.item_order!r}.")
        if not (isinstance(self.rho, numbers.Real) and self.rho >= 0):
            raise ValueError(f"rho must be a non-negative number; got {self.rho!r}.")
        for name in ("row_tolerance", "column_tolerance"):
            value = getattr(self, name)
            if not (isinstance(value, numbers.Real) and 0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1]; got {value!r}.")
        if not (isinstance(self.n_rounds, numbers.Integral) and self.n_rounds >= 0):
            raise ValueError(f"n_rounds must be a non-negative int; got {self.n_rounds!r}.")
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
        elif self.algorithm == "grecond":
            W, H = grecond(X, self.n_components, float(self.coverage))
        else:
            seed = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
            W, H = panda(X, self.n_components, self.cost, float(self.rho),
                         float(self.row_tolerance), float(self.column_tolerance),
                         self.item_order, int(self.n_rounds), seed, float(self.coverage))
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
        elif self.algorithm == "grecond":
            W = grecond_usage(X, H)
        else:
            W = asso_usage(X, H)
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
