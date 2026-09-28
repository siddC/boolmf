"""Align components across chains or fits (Hungarian matching on Jaccard similarity)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["jaccard_matrix", "match_components"]


def jaccard_matrix(A, B):
    """Pairwise Jaccard similarity between the rows of two boolean matrices.

    Parameters
    ----------
    A : array-like of shape (n_a, n_features)
    B : array-like of shape (n_b, n_features)

    Returns
    -------
    ndarray of shape (n_a, n_b)
    """
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)
    inter = Af @ Bf.T
    union = Af.sum(1)[:, None] + Bf.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J.astype(float)


def match_components(reference, other, threshold=0.5):
    """Match rows of ``other`` to rows of ``reference`` one-to-one.

    Uses the Hungarian algorithm on cost 1 - Jaccard and keeps pairs with Jaccard at least
    ``threshold``.

    Returns
    -------
    list of (reference_index, other_index, jaccard)
    """
    reference = np.asarray(reference, dtype=bool)
    other = np.asarray(other, dtype=bool)
    if reference.shape[0] == 0 or other.shape[0] == 0:
        return []
    J = jaccard_matrix(reference, other)
    r, c = linear_sum_assignment(1.0 - J)
    return [(int(i), int(j), float(J[i, j])) for i, j in zip(r, c) if J[i, j] >= threshold]
