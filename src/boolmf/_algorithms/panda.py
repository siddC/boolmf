"""PANDA+: Lucchese, Orlando & Perego (2014), A Unifying Framework for Mining Approximate Top-k
Binary Patterns, IEEE TKDE 26(12): 2900-2913, Algorithms 1-3, implemented as the paper writes
them.

A pattern is a set of transactions (rows) T and a set of items (columns) I; the patterns'
union approximates D. Each pattern starts as a noise-free core (FIND-CORE: items taken in a
score order, each kept when it does not raise the cost, the transactions shrinking to those
that still hold every item in the residual data) and is then extended (EXTEND-CORE: one pass
adding transactions, then items from the list the core rejected, one at a time, while the cost
does not rise and every item is in at least (1 - eps_c)|T| of the pattern's transactions and
every transaction holds at least (1 - eps_r)|I| of its items). A pattern is kept unless it
raises the cost; the residual data lose the cells it covers.

Cost functions (Table 1), with N the noise (cells where the union of patterns differs from D):

* ``"ja"``: |N|;
* ``"jp"``: rho * sum(|T| + |I|) + |N| (rho = 1 is PANDA's J_P);
* ``"je"``: the Typed XOR encoding of Miettinen & Vreeken (2011, KDD, Eqs. 7, 8, 12), which the
  paper cites: each pattern costs enc(n, |T|) + enc(m, |I|), and the noise
  enc(A, |E-|) + enc(nm - A, |E+|) with A the covered area, E- the covered zeros and E+ the
  uncovered ones, where enc(l, x) = log2 l + l H(x / l) (0 when l = 0). The terms for n, m and
  k are constant for a dataset and are left out.

Item orders (Section 3.6): ``"frequency"`` (support in the residual data), ``"couples"``
(sum of the supports of the pairs an item forms) and ``"correlation"`` (support of the
current core plus the item, recomputed whenever the core grows).

Choices the paper leaves open: ties in the item order keep the column order; E is a
first-in-first-out list; without a limit on the number of patterns PANDA+ also stops when a
pattern covers no uncovered ones. Randomization (Section 3.7) is only sketched in the paper;
as in the authors' code, round r = 0 .. R - 1 scores each item u * (s / s_max)^(2 / 0.9^r) with
u uniform on (0, 1), and the candidate pattern with the lowest cost is kept.
"""

import math

import numpy as np
from numba import njit

from .._sampler.kernels import _next_uniform

COSTS = {"ja": 0, "jp": 1, "je": 2}
ORDERS = {"frequency": 0, "couples": 1, "correlation": 2}


@njit(cache=True)
def _enc(l, x):
    if l <= 0:
        return 0.0
    out = math.log2(l)
    if 0 < x < l:
        p = x / l
        out += l * (-p * math.log2(p) - (1 - p) * math.log2(1 - p))
    return out


@njit(cache=True)
def _pattern_cost(kind, n, m, nt, ni):
    if kind == 2:
        return _enc(n, nt) + _enc(m, ni)
    return float(nt + ni)


@njit(cache=True)
def _cost(kind, rho, n, m, pc, area, fp, fn):
    if kind == 0:
        return float(fp + fn)
    if kind == 1:
        return rho * pc + fp + fn
    return pc + _enc(area, fp) + _enc(n * m - area, fn)


@njit(cache=True)
def _sorted_items(score, active, randomize, temp, state):
    """Active items by decreasing (possibly randomized) score; ties by index."""
    m = score.shape[0]
    idx = np.empty(m, np.int64)
    key = np.empty(m, np.float64)
    c = 0
    smax = 0.0
    for j in range(m):
        if active[j] and score[j] > smax:
            smax = score[j]
    for j in range(m):
        if active[j]:
            s = score[j]
            if randomize:
                state, u = _next_uniform(state)
                s = u * (s / smax) ** temp if smax > 0 else 0.0
            idx[c] = j
            key[c] = -s
            c += 1
    order = np.argsort(key[:c], kind="mergesort")
    return idx[:c][order], state


@njit(cache=True)
def _find_core(D, R, kind, rho, pc, area, fp, fn, order_kind, randomize, temp, state):
    n, m = D.shape
    freq = np.zeros(m)
    for i in range(n):
        for j in range(m):
            if R[i, j]:
                freq[j] += 1
    active = freq > 0
    ext = np.empty(m, np.int64)
    n_ext = 0
    T = np.zeros(n, np.bool_)
    I = np.zeros(m, np.bool_)
    if not active.any():
        return T, I, ext[:0], state
    if order_kind == 1:                                  # couples frequency
        score = np.zeros(m)
        for i in range(n):
            r = 0
            for j in range(m):
                r += R[i, j]
            for j in range(m):
                if R[i, j]:
                    score[j] += r - 1
    else:
        score = freq.copy()
    order, state = _sorted_items(score, active, randomize, temp, state)
    s1 = order[0]
    I[s1] = True
    nt = 0
    for i in range(n):
        if R[i, s1]:
            T[i] = True
            nt += 1
    ni = 1
    cur = _cost(kind, rho, n, m, pc + _pattern_cost(kind, n, m, nt, ni), area + nt, fp,
                fn - nt)
    done = np.zeros(m, np.bool_)
    done[s1] = True
    pos = 1
    resort = order_kind == 2
    while True:
        if resort:                                        # correlation: re-rank the rest
            sc = np.zeros(m)
            for i in range(n):
                if T[i]:
                    for j in range(m):
                        if R[i, j]:
                            sc[j] += 1
            order, state = _sorted_items(sc, active & ~done, randomize, temp, state)
            pos = 0
            resort = False
        if pos >= order.shape[0]:
            break
        h = order[pos]
        if done[h]:
            pos += 1
            continue
        pos += 1
        done[h] = True
        nt2 = 0
        for i in range(n):
            if T[i] and R[i, h]:
                nt2 += 1
        a = nt2 * (ni + 1)
        new = _cost(kind, rho, n, m, pc + _pattern_cost(kind, n, m, nt2, ni + 1), area + a,
                    fp, fn - a)
        if new <= cur:
            for i in range(n):
                if T[i] and not R[i, h]:
                    T[i] = False
            I[h] = True
            nt, ni, cur = nt2, ni + 1, new
            resort = order_kind == 2
        else:
            ext[n_ext] = h
            n_ext += 1
    return T, I, ext[:n_ext], state


@njit(cache=True)
def _extend_core(D, cov, T, I, ext, kind, rho, pc, area, fp, fn, eps_r, eps_c):
    n, m = D.shape
    nt = 0
    for i in range(n):
        nt += T[i]
    ni = 0
    for j in range(m):
        ni += I[j]
    # zeros of D per pattern row (over I) and per pattern column (over T); new cells of the
    # pattern (not covered by earlier patterns) that are ones / zeros of D
    row0 = np.zeros(n, np.int64)
    col0 = np.zeros(m, np.int64)
    new1 = 0
    new0 = 0
    for i in range(n):
        if T[i]:
            for j in range(m):
                if I[j]:
                    if D[i, j] == 0:
                        row0[i] += 1
                        col0[j] += 1
                    if not cov[i, j]:
                        if D[i, j]:
                            new1 += 1
                        else:
                            new0 += 1
    cur = _cost(kind, rho, n, m, pc + _pattern_cost(kind, n, m, nt, ni), area + new1 + new0,
                fp + new0, fn - new1)
    head = 0
    added_item = True
    while added_item:
        for i in range(n):                                # add transactions
            if T[i]:
                continue
            z = 0
            d1 = 0
            d0 = 0
            for j in range(m):
                if I[j]:
                    if D[i, j] == 0:
                        z += 1
                    if not cov[i, j]:
                        if D[i, j]:
                            d1 += 1
                        else:
                            d0 += 1
            if z > eps_r * ni:
                continue
            ok = True
            for j in range(m):                            # every item keeps enough support
                if I[j]:
                    zj = col0[j] + (D[i, j] == 0)
                    if zj > eps_c * (nt + 1):
                        ok = False
                        break
            if not ok:
                continue
            new = _cost(kind, rho, n, m, pc + _pattern_cost(kind, n, m, nt + 1, ni),
                        area + new1 + new0 + d1 + d0, fp + new0 + d0, fn - new1 - d1)
            if new <= cur:
                T[i] = True
                nt += 1
                row0[i] = z
                for j in range(m):
                    if I[j] and D[i, j] == 0:
                        col0[j] += 1
                new1 += d1
                new0 += d0
                cur = new
        added_item = False
        while head < ext.shape[0]:                        # add one item from E
            e = ext[head]
            head += 1
            z = 0
            d1 = 0
            d0 = 0
            for i in range(n):
                if T[i]:
                    if D[i, e] == 0:
                        z += 1
                    if not cov[i, e]:
                        if D[i, e]:
                            d1 += 1
                        else:
                            d0 += 1
            if z > eps_c * nt:
                continue
            ok = True
            for i in range(n):                            # every transaction keeps enough items
                if T[i]:
                    zi = row0[i] + (D[i, e] == 0)
                    if zi > eps_r * (ni + 1):
                        ok = False
                        break
            if not ok:
                continue
            new = _cost(kind, rho, n, m, pc + _pattern_cost(kind, n, m, nt, ni + 1),
                        area + new1 + new0 + d1 + d0, fp + new0 + d0, fn - new1 - d1)
            if new <= cur:
                I[e] = True
                ni += 1
                col0[e] = z
                for i in range(n):
                    if T[i] and D[i, e] == 0:
                        row0[i] += 1
                new1 += d1
                new0 += d0
                cur = new
                added_item = True
                break
    return T, I, cur, new1, new0


def panda(X, n_components=None, cost="je", rho=1.0, eps_r=1.0, eps_c=1.0,
          order="correlation", n_rounds=0, random_state=None, coverage=1.0):
    """PANDA+ (Algorithm 1). ``n_rounds`` > 0 runs that many randomized rounds per pattern.

    Returns
    -------
    usage : ndarray of bool, shape (n_samples, k)
    basis : ndarray of bool, shape (k, n_features)
    """
    D = np.ascontiguousarray(np.asarray(X, bool)).astype(np.uint8)
    n, m = D.shape
    kind, order_kind = COSTS[cost], ORDERS[order]
    rng = np.random.default_rng(random_state)
    cov = np.zeros((n, m), np.bool_)
    R = D.astype(np.bool_)
    pc, area, fp, fn = 0.0, 0, 0, int(D.sum())
    usage, basis = [], []
    while n_components is None or len(basis) < n_components:
        base = _cost(kind, rho, n, m, pc, area, fp, fn)
        best = None
        rounds = range(n_rounds) if n_rounds > 0 else [None]
        for r in rounds:
            randomize = r is not None
            temp = 2.0 / 0.9 ** r if randomize else 1.0
            state = np.uint64(rng.integers(2**63 - 1))
            T, I, ext, _ = _find_core(D, R, kind, rho, pc, area, fp, fn, order_kind, randomize,
                                      temp, state)
            if not I.any():
                continue
            T, I, c, new1, new0 = _extend_core(D, cov, T, I, ext, kind, rho, pc, area, fp, fn,
                                               float(eps_r), float(eps_c))
            if best is None or c < best[2]:
                best = (T.copy(), I.copy(), c, new1, new0)
        if best is None:
            break
        T, I, c, new1, new0 = best
        if base < c:
            break
        usage.append(T)
        basis.append(I)
        pc += _pattern_cost(kind, n, m, int(T.sum()), int(I.sum()))
        area += new1 + new0
        fp += new0
        fn -= new1
        cov[np.ix_(T, I)] = True
        R[np.ix_(T, I)] = False
        if n_components is None and new1 == 0:
            break
        if coverage < 1 and 1 - fn / max(1, int(D.sum())) >= coverage:
            break
    k = len(basis)
    return (np.array(usage).T.reshape(n, k), np.array(basis).reshape(k, m))


def panda_cost(X, usage, basis, cost="je", rho=1.0):
    """The cost J of a pattern set on X (for model order selection and checks)."""
    D = np.asarray(X, bool)
    n, m = D.shape
    kind = COSTS[cost]
    covered = (usage.astype(np.int64) @ basis.astype(np.int64)) > 0
    pc = sum(_pattern_cost(kind, n, m, int(usage[:, k].sum()), int(basis[k].sum()))
             for k in range(basis.shape[0]))
    area = int(covered.sum())
    fp = int((covered & ~D).sum())
    fn = int((D & ~covered).sum())
    return _cost(kind, rho, n, m, float(pc), area, fp, fn)
