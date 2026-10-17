"""Collapsed Indian buffet process sampler with explicit births of new components.

Used with ``births="enumerate"`` (Wood, Griffiths & Ghahramani 2006; Rukat & Yau 2019) and
``births="metropolis"`` (Meeds, Ghahramani, Neal & Roweis 2007). The IBP side (here the rows,
Z) has its component probabilities integrated out, so each row is updated in turn given all
the others:

* an existing component k with m_-i,k other carriers is kept by row i with prior probability
  m_-i,k / N;
* ``enumerate``: after the shared components, those that only row i carries are dropped and
  the number of new ones is drawn from its exact conditional, Poisson(alpha / N) times the
  likelihood of row i with the new components' memberships (independent Bernoulli(p)) summed
  out, evaluated for 0 .. max_new new components (Wood et al. 2006, Eqs. 13-15). The new
  memberships are then drawn from their exact joint conditional given row i (one feature at a
  time: the number of new components covering the feature, then which ones), or, with
  ``exact_members=False``, set by one Gibbs pass from zero as the papers write it;
* ``metropolis``: the components only row i carries are proposed to be replaced by
  Poisson(alpha / N) new ones with memberships drawn from their prior; the proposal is accepted
  with the likelihood ratio of row i (Meeds et al. 2007).

The other side (U) has one membership probability p shared by all components and is updated
by the usual parallel Gibbs kernel. Counts C and the likelihood tables are as in ``kernels``;
the tables must run to count K + max_new + 1.
"""

import math

import numpy as np
from numba import njit

from .kernels import _decide, _next_uniform, _row_state

BIRTH_ENUMERATE, BIRTH_METROPOLIS = 0, 1


@njit(inline="always")
def _ll(v, c, T1, T0):
    return T1[c] if v == 1 else T0[c]


@njit(cache=True)
def _log_poisson(k, mean):
    return k * math.log(mean) - mean - math.lgamma(k + 1.0)


@njit(cache=True)
def _log_binom_pmf(j, n, p):
    return (math.lgamma(n + 1.0) - math.lgamma(j + 1.0) - math.lgamma(n - j + 1.0)
            + j * math.log(p) + (n - j) * math.log1p(-p))


@njit(cache=True)
def _logsumexp(x, n):
    mx = -np.inf
    for i in range(n):
        if x[i] > mx:
            mx = x[i]
    if mx == -np.inf:
        return mx
    s = 0.0
    for i in range(n):
        s += math.exp(x[i] - mx)
    return mx + math.log(s)


@njit(cache=True)
def _pick_index(state, logw, n):
    """Draw an index with probabilities proportional to exp(logw[:n])."""
    lz = _logsumexp(logw, n)
    state, u = _next_uniform(state)
    acc = 0.0
    for i in range(n):
        acc += math.exp(logw[i] - lz)
        if u < acc:
            return state, i
    return state, n - 1


@njit(cache=True)
def _free_slot(m, used, K):
    for k in range(K):
        if not used[k]:
            return k
    return -1


@njit(cache=True)
def collapsed_rows(V, Z, U, C, T1, T0, alpha, p, births, max_new, metropolis, seed, stats,
                   exact_members=True):
    """One pass over the rows of Z (sequential), with births; U columns of new or removed
    components are set here. stats[0] counts births, stats[1] deaths, stats[2] rows that wanted
    more new components than free slots (the truncation was hit)."""
    n, F = V.shape
    K = Z.shape[1]
    m = np.zeros(K, np.int64)
    used = np.zeros(K, np.bool_)
    for k in range(K):
        for i in range(n):
            m[k] += Z[i, k]
        used[k] = m[k] > 0
    logw = np.empty(max_new + 1)
    lj = np.empty(max_new + 1)
    ll_new = np.empty(max_new + 1)
    hist = np.zeros((2, K + 1), np.int64)
    cum = np.empty((2, K + 1, max_new + 1))
    lbin = np.full((max_new + 1, max_new + 1), -np.inf)     # log Binomial(j; kn, p)
    cum_prior = np.empty((max_new + 1, max_new + 1))
    for kn in range(max_new + 1):
        acc = 0.0
        for j in range(kn + 1):
            lbin[kn, j] = _log_binom_pmf(j, kn, p)
            acc += math.exp(lbin[kn, j])
            cum_prior[kn, j] = acc
    for i in range(n):
        state = _row_state(seed, i)
        # ---- existing components ------------------------------------------------------
        for k in range(K):
            if not used[k]:
                continue
            old = Z[i, k]
            m_minus = m[k] - old
            if m_minus == 0:
                continue                                # row i's singletons: births below
            lo = math.log(m_minus) - math.log(n - m_minus)
            for t in range(F):
                if U[t, k] == 0:
                    continue
                v = V[i, t]
                if v < 0:
                    continue
                cm = C[i, t] - old
                lo += _ll(v, cm + 1, T1, T0) - _ll(v, cm, T1, T0)
            state, new = _decide(state, lo, old, metropolis)
            if new != old:
                for t in range(F):
                    if U[t, k] != 0:
                        C[i, t] += new - old
                Z[i, k] = new
                m[k] += new - old
        # ---- births -------------------------------------------------------------------
        mean_new = alpha / n
        if births == BIRTH_ENUMERATE:
            # drop row i's singletons (kept in place while the shared components were updated,
            # so each of those updates saw the full state); their number and memberships are
            # redrawn below from their joint conditional
            for k in range(K):
                if used[k] and Z[i, k] == 1 and m[k] == 1:
                    Z[i, k] = 0
                    for t in range(F):
                        if U[t, k] != 0:
                            C[i, t] -= 1
                            U[t, k] = 0
                    m[k] = 0
                    used[k] = False
                    stats[1] += 1
            # the likelihood of row i depends on each feature only through (x, count), so the
            # sum over features is a sum over the histogram of (x, count) pairs
            hist[:, :] = 0
            for t in range(F):
                v = V[i, t]
                if v >= 0:
                    hist[v, C[i, t]] += 1
            for kn in range(max_new + 1):
                ll_new[kn] = 0.0
            for v in range(2):
                for c in range(K + 1):
                    h = hist[v, c]
                    if h == 0:
                        continue
                    for kn in range(max_new + 1):
                        for j in range(kn + 1):
                            lj[j] = lbin[kn, j] + _ll(v, c + j, T1, T0)
                        ll_new[kn] += h * _logsumexp(lj, kn + 1)
            for kn in range(max_new + 1):
                logw[kn] = _log_poisson(kn, mean_new) + ll_new[kn]
            state, kn = _pick_index(state, logw, max_new + 1)
            if kn == 0:
                continue
            slots = np.empty(kn, np.int64)
            got = 0
            for _ in range(kn):
                s = _free_slot(m, used, K)
                if s < 0:
                    stats[2] += 1
                    break
                slots[got] = s
                used[s] = True
                m[s] = 1
                Z[i, s] = 1
                got += 1
            stats[0] += got
            if got == 0:
                continue
            if not exact_members:
                # as written in Wood et al. (2006) and Rukat & Yau (2019): the new memberships
                # start at 0 and get one Gibbs pass, component by component. Exact for one new
                # component; with several, the pass is not a draw from their joint conditional
                lp = math.log(p) - math.log1p(-p)
                for q in range(got):
                    s = slots[q]
                    for t in range(F):
                        v = V[i, t]
                        lo = lp
                        if v >= 0:
                            c = C[i, t]
                            lo += _ll(v, c + 1, T1, T0) - _ll(v, c, T1, T0)
                        state, new = _decide(state, lo, 0, False)
                        if new == 1:
                            U[t, s] = 1
                            C[i, t] += 1
                continue
            # memberships of the new components, exact joint draw feature by feature: the
            # number j of new components containing the feature (weights depend only on
            # (x, count), so cumulative weights are tabulated once), then which ones
            for v in range(2):
                for c in range(K + 1):
                    if hist[v, c] == 0:
                        continue
                    for j in range(got + 1):
                        lj[j] = lbin[got, j] + _ll(v, c + j, T1, T0)
                    lz = _logsumexp(lj, got + 1)
                    acc = 0.0
                    for j in range(got + 1):
                        acc += math.exp(lj[j] - lz)
                        cum[v, c, j] = acc
            for t in range(F):
                v = V[i, t]
                c = C[i, t]
                state, u = _next_uniform(state)
                nj = 0
                if v < 0:                       # unobserved: memberships from the prior
                    while nj < got and u > cum_prior[got, nj]:
                        nj += 1
                else:
                    while nj < got and u > cum[v, c, nj]:
                        nj += 1
                chosen = 0
                for q in range(got):
                    state, u = _next_uniform(state)
                    if u * (got - q) < nj - chosen:
                        U[t, slots[q]] = 1
                        chosen += 1
                C[i, t] += nj
        else:
            # current singletons of row i
            n_single = 0
            single = np.empty(K, np.int64)
            for k in range(K):
                if used[k] and Z[i, k] == 1 and m[k] == 1:
                    single[n_single] = k
                    n_single += 1
            # proposal: Poisson(alpha / n) new components, memberships from the prior
            state, u = _next_uniform(state)
            kn = 0
            cdf = math.exp(-mean_new)
            pk = cdf
            while u > cdf and kn < max_new:
                kn += 1
                pk *= mean_new / kn
                cdf += pk
            cnt_single = np.zeros(F, np.int64)
            for q in range(n_single):
                for t in range(F):
                    cnt_single[t] += U[t, single[q]]
            prop = np.zeros((F, max(kn, 1)), np.int8)
            for q in range(kn):
                for t in range(F):
                    state, u = _next_uniform(state)
                    if u < p:
                        prop[t, q] = 1
            dll = 0.0
            for t in range(F):
                v = V[i, t]
                if v < 0:
                    continue
                c0 = C[i, t] - cnt_single[t]
                cn = c0
                for q in range(kn):
                    cn += prop[t, q]
                dll += _ll(v, cn, T1, T0) - _ll(v, C[i, t], T1, T0)
            state, u = _next_uniform(state)
            if not (dll >= 0 or u < math.exp(dll)):
                continue
            n_free = 0
            for k in range(K):
                if not used[k]:
                    n_free += 1
            if kn > n_free + n_single:
                stats[2] += 1
                continue
            for q in range(n_single):                   # deaths
                k = single[q]
                Z[i, k] = 0
                for t in range(F):
                    if U[t, k] != 0:
                        C[i, t] -= 1
                        U[t, k] = 0
                m[k] = 0
                used[k] = False
            stats[1] += n_single
            for q in range(kn):                         # births
                s = _free_slot(m, used, K)
                used[s] = True
                m[s] = 1
                Z[i, s] = 1
                for t in range(F):
                    if prop[t, q] != 0:
                        U[t, s] = 1
                        C[i, t] += 1
            stats[0] += kn
    return 0
