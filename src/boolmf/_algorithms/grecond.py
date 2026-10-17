"""GreConD: Belohlavek & Vychodil (2010), Discovery of optimal factors in binary data via a
novel method of matrix decomposition, J. Comput. Syst. Sci. 76: 3-20, Algorithm 2.

Factors are formal concepts (C, D): D is a set of features, C = D-down the samples that have
all of them, and D = C-up. Each factor is grown from the empty set of features by adding the
feature whose closure covers the most still-uncovered ones, until no feature helps; its cells
are then marked covered. Factors never cover a 0 ("from below").

Details follow the authors' reference implementation (Trnecka, GitHub
martin-trnecka/matrix-factorization-algorithms, GreConD.m), with which the published factor
counts are reproduced: only features with uncovered ones are tried, ties go to the lowest
feature index, and growth stops when a pass leaves the feature set unchanged.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _next_factor(M, U, tried):
    n, m = M.shape
    v = 0
    d = np.zeros(m, np.bool_)
    d_old = np.zeros(m, np.bool_)
    d_mid = np.zeros(m, np.bool_)
    e = np.ones(n, np.bool_)
    c = np.zeros(n, np.bool_)
    rows = np.empty(n, np.int64)
    b = np.zeros(m, np.bool_)
    while True:
        for j in range(m):
            if not tried[j] or d[j]:
                continue
            sa = 0
            for i in range(n):
                if e[i] and M[i, j]:
                    rows[sa] = i
                    sa += 1
            if sa * m <= v:
                continue
            sb = 0
            for t in range(m):
                ok = True
                for r in range(sa):
                    if not M[rows[r], t]:
                        ok = False
                        break
                b[t] = ok
                if ok:
                    sb += 1
            if sa * sb <= v:
                continue
            cost = 0
            for r in range(sa):
                i = rows[r]
                for t in range(m):
                    if b[t] and U[i, t]:
                        cost += 1
            if cost > v:
                v = cost
                d_mid[:] = b
                c[:] = False
                for r in range(sa):
                    c[rows[r]] = True
        d[:] = d_mid
        e[:] = c
        same = True
        for t in range(m):
            if d[t] != d_old[t]:
                same = False
                break
        if same:
            break
        d_old[:] = d
    return c, d.copy(), v


def grecond(X, n_components=None, coverage=1.0):
    """Greedy concept factorization; stops at ``n_components`` factors, at the given fraction
    of ones covered, or at an exact cover.

    Returns
    -------
    usage : ndarray of bool, shape (n_samples, k)
    basis : ndarray of bool, shape (k, n_features)
    """
    M = np.ascontiguousarray(np.asarray(X, bool))
    n, m = M.shape
    U = M.copy()
    total = int(M.sum())
    left = total
    usage, basis = [], []
    while left > 0 and (n_components is None or len(basis) < n_components):
        c, d, _ = _next_factor(M, U, U.any(axis=0))
        usage.append(c)
        basis.append(d)
        left -= int(U[np.ix_(c, d)].sum())
        U[np.ix_(c, d)] = False
        if coverage < 1 and (total - left) / total >= coverage:
            break
    k = len(basis)
    return (np.array(usage).T.reshape(n, k), np.array(basis).reshape(k, m))


def grecond_usage(X, basis):
    """Usage from below: a sample uses a factor when it has all of the factor's features."""
    X = np.asarray(X, bool)
    usage = np.zeros((X.shape[0], basis.shape[0]), bool)
    for l in range(basis.shape[0]):
        usage[:, l] = X[:, basis[l]].all(axis=1)
    return usage
