"""MEBF: Wan, Chang, Zhao, Li, Cao & Zhang (2020), Fast and Efficient Boolean Matrix
Factorization by Geometric Segmentation, AAAI 34: 6086-6093 (arXiv:1909.03991), Algorithms 1-3,
implemented as the paper writes them.

Each step reorders the residual matrix to be upper-triangular-like (UTL: row sums
non-increasing from the top, column sums non-decreasing from the left, empty rows and columns
dropped) and grows a pattern from its median column (the rows of X whose similarity
<x_j, d> / <d, d> to that column d exceeds t) or from its median row, keeping the one with the
lower cost |R XOR (a b)| on the residual R (bidirectional growth). A pattern is kept when it
does not raise the cost |X XOR (A B)| on the original matrix; otherwise the step is retried
with a pattern grown from the intersection of the two fullest columns or of the two fullest
rows (weak signal detection), and MEBF stops if that also raises the cost. The residual loses
the ones each kept pattern covers.

Choices the paper leaves open: sorts are stable (ties keep the original order), the median of
an even number of columns or rows is the lower middle one, a candidate with an empty seed is
skipped, and without a limit on the number of patterns MEBF also stops when the residual is
empty or a kept pattern covers none of its ones. The authors' R code (github.com/clwan/MEBF)
differs from the paper in its seeds, expansion threshold, selection score, weak-signal rule and
stopping rule; it is not followed here.
"""

import numpy as np


def _cost(R, a, b):
    """|R XOR (a b)| for one rank-1 pattern."""
    inside = np.count_nonzero(R[np.ix_(a, b)])
    return int(np.count_nonzero(R)) - inside + (int(a.sum()) * int(b.sum()) - inside)


def _utl(R):
    """Rows (non-increasing sums) and columns (non-decreasing sums) of R that hold a one."""
    rs, cs = R.sum(1), R.sum(0)
    rows = np.flatnonzero(rs > 0)
    cols = np.flatnonzero(cs > 0)
    rows = rows[np.argsort(-rs[rows], kind="stable")]
    cols = cols[np.argsort(cs[cols], kind="stable")]
    return rows, cols


def _grow_from_column(R, d, t):
    """Pattern with rows d and every column j with <R_:j, d> / <d, d> > t."""
    size = int(d.sum())
    if size == 0:
        return None
    return d, (d.astype(np.int64) @ R) / size > t


def _grow_from_row(R, f, t):
    size = int(f.sum())
    if size == 0:
        return None
    return (R @ f.astype(np.int64)) / size > t, f


def _best(R, candidates):
    best, best_cost = None, None
    for c in candidates:
        if c is None:
            continue
        cost = _cost(R, *c)
        if best is None or cost < best_cost:
            best, best_cost = c, cost
    return best


def bidirectional_growth(R, t):
    rows, cols = _utl(R)
    if rows.size == 0:
        return None
    d = R[:, cols[(cols.size - 1) // 2]].copy()
    f = R[rows[(rows.size - 1) // 2], :].copy()
    # the column pattern wins ties, as in Algorithm 2 (the row pattern only if strictly cheaper)
    return _best(R, [_grow_from_column(R, d, t), _grow_from_row(R, f, t)])


def weak_signal_detection(R, t):
    rows, cols = _utl(R)
    candidates = []
    if cols.size >= 2:
        candidates.append(_grow_from_column(R, R[:, cols[-1]] & R[:, cols[-2]], t))
    if rows.size >= 2:
        candidates.append(_grow_from_row(R, R[rows[0], :] & R[rows[1], :], t))
    return _best(R, candidates)


def mebf(X, n_components=None, t=0.5, coverage=1.0):
    """MEBF factorization (Algorithm 1).

    Returns
    -------
    usage : ndarray of bool, shape (n_samples, k)
    basis : ndarray of bool, shape (k, n_features)
    """
    X = np.asarray(X, bool)
    n, m = X.shape
    R = X.copy()
    total = max(1, int(X.sum()))
    covered = np.zeros((n, m), bool)
    cost0 = None
    usage, basis = [], []
    while n_components is None or len(basis) < n_components:
        best = None
        for step in (bidirectional_growth, weak_signal_detection):
            cand = step(R, t)
            if cand is None:
                continue
            a, b = cand
            cost = int(np.count_nonzero(X != (covered | np.outer(a, b))))
            if cost0 is None or cost <= cost0:
                best = (a, b, cost)
                break
        if best is None:
            break
        a, b, cost0 = best
        new = R[np.ix_(a, b)].any()
        usage.append(a)
        basis.append(b)
        covered |= np.outer(a, b)
        R[np.ix_(a, b)] = False
        if n_components is None and (not new or not R.any()):
            break
        if coverage < 1 and np.count_nonzero(covered & X) / total >= coverage:
            break
    k = len(basis)
    return (np.array(usage).T.reshape(n, k), np.array(basis).reshape(k, m))
