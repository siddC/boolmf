"""Asso: Miettinen, Mielikäinen, Gionis, Das & Mannila (2008), The Discrete Basis Problem,
IEEE TKDE 20(10): 1348-1362, Algorithm 1.

Candidate basis vectors are the rows of the association matrix: row i holds the features j
with confidence c(i => j) = |x_i AND x_j| / |x_i| >= tau, over the columns x_i of X. Basis
vectors are chosen greedily; each maximizes the cover function
w+ |{X = 1 covered}| - w- |{X = 0 covered}| given the ones already chosen, and each sample
uses it exactly when that raises its own cover.

Choices the paper leaves open, as in Miettinen's reference code (``basso``): a sample uses a
basis vector only when the gain is strictly positive, and ties between candidates go to the
lowest feature index. A feature with no ones gives an empty candidate.
"""

import numpy as np


def association_matrix(X, tau):
    """Boolean (n_features, n_features) matrix: A[i, j] = c(i => j) >= tau."""
    Xf = X.astype(np.float64)
    both = Xf.T @ Xf                                     # |x_i AND x_j|
    size = np.diag(both).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = both / size[:, None]
    A = conf >= tau
    A[size == 0] = False
    return A


def _gains(X, covered, B, w_pos, w_neg):
    """Per-sample gain of each candidate row of B: (n_samples, n_candidates)."""
    free = ~covered
    hit_one = (X & free).astype(np.float64) @ B.T.astype(np.float64)
    hit_zero = (~X & free).astype(np.float64) @ B.T.astype(np.float64)
    return w_pos * hit_one - w_neg * hit_zero


def asso(X, n_components=None, tau=0.5, w_pos=1.0, w_neg=1.0, coverage=1.0):
    """Greedy Asso factorization.

    ``n_components=None`` stops when no candidate raises the cover function (an extension:
    the paper fixes k); ``coverage`` < 1 stops once that fraction of the ones is covered.
    With a fixed ``n_components`` a basis vector that no sample uses is still added, as in
    the paper.

    Returns
    -------
    usage : ndarray of bool, shape (n_samples, k)
    basis : ndarray of bool, shape (k, n_features)
    """
    X = np.asarray(X, bool)
    n, m = X.shape
    cand = association_matrix(X, tau)
    covered = np.zeros((n, m), bool)
    total = max(1, int(X.sum()))
    usage, basis = [], []
    while n_components is None or len(basis) < n_components:
        G = _gains(X, covered, cand, w_pos, w_neg)
        score = np.where(G > 0, G, 0.0).sum(0)
        best = int(np.argmax(score))                      # first maximum
        if n_components is None and score[best] <= 0:
            break
        s = G[:, best] > 0
        usage.append(s)
        basis.append(cand[best].copy())
        covered |= np.outer(s, cand[best])
        if coverage < 1 and np.count_nonzero(covered & X) / total >= coverage:
            break
    k = len(basis)
    return (np.array(usage).T.reshape(n, k), np.array(basis).reshape(k, m))


def asso_usage(X, basis, w_pos=1.0, w_neg=1.0):
    """Usage of fixed basis vectors, in order, by the rule Asso uses while fitting."""
    X = np.asarray(X, bool)
    covered = np.zeros(X.shape, bool)
    usage = np.zeros((X.shape[0], basis.shape[0]), bool)
    for l in range(basis.shape[0]):
        s = _gains(X, covered, basis[l:l + 1], w_pos, w_neg)[:, 0] > 0
        usage[:, l] = s
        covered |= np.outer(s, basis[l])
    return usage
