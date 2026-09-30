"""Numba kernels for the BoolMF Gibbs sampler.

Conventions
-----------
V : int8 (n_samples, n_features)     1 present, 0 absent, -1 missing
Z : int8 (n_samples, K)              activation of component k in sample i
U : int8 (n_features, K)             membership of feature j in component k
C : int16 (n_samples, n_features)    number of active components containing feature j in sample i
T1, T0 : float64 (n_rows, K + 2)     log P(x_ij = 1 | c) and log P(x_ij = 0 | c) for c = 0 .. K + 1:
                                     one row per sample, or a single row shared by all samples
                                     when the rates are global (row index i * rs, rs = 0 or 1)

Both likelihoods and every rate model enter the kernels only through the tables T1 and T0, so
one kernel serves ``likelihood="noisy_or"`` and ``"or_flip"``, with global or per-sample rates.
``project_activations`` (new samples) takes a single row of each table.

Randomness uses a counter-based SplitMix64 stream per (sweep seed, row), so results do not depend
on the number of threads or on how numba schedules them.
"""

import numpy as np
from numba import njit, prange

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0


@njit(inline="always")
def _next_uniform(state):
    """SplitMix64 step: returns (new_state, uniform in [0, 1))."""
    state = state + _GOLDEN
    z = state
    z = (z ^ (z >> _S30)) * _MIX1
    z = (z ^ (z >> _S27)) * _MIX2
    z = z ^ (z >> _S31)
    return state, (z >> _S11) * _INV53


@njit(inline="always")
def _row_state(seed, row):
    s = np.uint64(seed) ^ (np.uint64(row + 1) * _GOLDEN)
    s, _ = _next_uniform(s)
    return s


@njit(inline="always")
def _decide(state, lo, old, metropolis):
    """New value of a binary variable whose log-odds of being 1 given the rest is ``lo``.

    Gibbs: draw from the conditional. Metropolised Gibbs (Liu 1996): always propose the other
    value and accept with probability min(1, p(other) / p(current)).
    """
    state, u = _next_uniform(state)
    if metropolis:
        r = lo if old == 0 else -lo
        if r >= 0.0 or u < np.exp(r):
            return state, np.int8(1 - old)
        return state, np.int8(old)
    return state, (np.int8(1) if u * (1.0 + np.exp(-lo)) < 1.0 else np.int8(0))


@njit(cache=True)
def counts_from_state(Z, U):
    """C[i, j] = sum_k Z[i, k] * U[j, k]."""
    n, K = Z.shape
    F = U.shape[0]
    C = np.zeros((n, F), np.int16)
    for i in prange(n):
        for k in range(K):
            if Z[i, k] == 0:
                continue
            for j in range(F):
                if U[j, k] != 0:
                    C[i, j] += 1
    return C


@njit(parallel=True, cache=True)
def update_memberships(V, U, C, T1, T0, logit_rho, act_ptr, act_idx, update_mask, seed,
                       metropolis=False):
    """Gibbs update of U[j, k] for every feature j (in parallel) and component k.

    Rows of U are independent given Z, so updating all features at once is an exact
    Gibbs step. Only samples where component k is active carry likelihood information.
    """
    F, K = U.shape
    rs = 1 if T1.shape[0] > 1 else 0
    pr = 1 if logit_rho.shape[0] > 1 else 0
    for j in prange(F):
        state = _row_state(seed, j)
        for k in range(K):
            if not update_mask[k]:
                continue
            lo = logit_rho[j * pr, k]
            old = U[j, k]
            for t in range(act_ptr[k], act_ptr[k + 1]):
                i = act_idx[t]
                v = V[i, j]
                if v < 0:
                    continue
                cm = C[i, j] - old
                ti = i * rs
                if v == 1:
                    lo += T1[ti, cm + 1] - T1[ti, cm]
                else:
                    lo += T0[ti, cm + 1] - T0[ti, cm]
            state, new = _decide(state, lo, old, metropolis)
            if new != old:
                d = np.int16(new - old)
                for t in range(act_ptr[k], act_ptr[k + 1]):
                    C[act_idx[t], j] += d
                U[j, k] = new


@njit(parallel=True, cache=True)
def update_activations(V, Z, C, T1, T0, logit_pi, mem_ptr, mem_idx, update_mask, seed,
                       metropolis=False):
    """Gibbs update of Z[i, k] for every sample i (in parallel) and component k."""
    n, K = Z.shape
    rs = 1 if T1.shape[0] > 1 else 0
    pz = 1 if logit_pi.shape[0] > 1 else 0
    for i in prange(n):
        state = _row_state(seed, i)
        ti = i * rs
        for k in range(K):
            if not update_mask[k]:
                continue
            lo = logit_pi[i * pz, k]
            old = Z[i, k]
            for t in range(mem_ptr[k], mem_ptr[k + 1]):
                j = mem_idx[t]
                v = V[i, j]
                if v < 0:
                    continue
                cm = C[i, j] - old
                if v == 1:
                    lo += T1[ti, cm + 1] - T1[ti, cm]
                else:
                    lo += T0[ti, cm + 1] - T0[ti, cm]
            state, new = _decide(state, lo, old, metropolis)
            if new != old:
                d = np.int16(new - old)
                for t in range(mem_ptr[k], mem_ptr[k + 1]):
                    C[i, mem_idx[t]] += d
                Z[i, k] = new


@njit(parallel=True, cache=True)
def project_activations(V, U, T1, T0, logit_pi, mem_ptr, mem_idx, row_seeds, n_sweeps, n_keep):
    """Sample activations for new samples with memberships held fixed.

    Each row gets its own random stream from ``row_seeds`` (derived from the row's content),
    so the result for a row does not depend on which other rows are projected with it.
    Returns the mean activation over the last ``n_keep`` sweeps.
    """
    n, F = V.shape
    K = U.shape[1]
    out = np.zeros((n, K), np.float64)
    for i in prange(n):
        state = np.uint64(row_seeds[i])
        z = np.zeros(K, np.int8)
        c = np.zeros(F, np.int16)
        for s in range(n_sweeps):
            for k in range(K):
                lo = logit_pi[k]
                old = z[k]
                for t in range(mem_ptr[k], mem_ptr[k + 1]):
                    j = mem_idx[t]
                    v = V[i, j]
                    if v < 0:
                        continue
                    cm = c[j] - old
                    if v == 1:
                        lo += T1[cm + 1] - T1[cm]
                    else:
                        lo += T0[cm + 1] - T0[cm]
                state, u = _next_uniform(state)
                new = np.int8(1) if u * (1.0 + np.exp(-lo)) < 1.0 else np.int8(0)
                if new != old:
                    d = np.int16(new - old)
                    for t in range(mem_ptr[k], mem_ptr[k + 1]):
                        c[mem_idx[t]] += d
                    z[k] = new
            if s >= n_sweeps - n_keep:
                for k in range(K):
                    out[i, k] += z[k]
        for k in range(K):
            out[i, k] /= n_keep
    return out


@njit(parallel=True, cache=True)
def count_histograms(V, C, cmax):
    """H1[c], H0[c]: number of observed present / absent entries with count c (capped at cmax)."""
    n, F = V.shape
    nb = (n + 63) // 64
    H1b = np.zeros((nb, cmax + 1), np.int64)
    H0b = np.zeros((nb, cmax + 1), np.int64)
    for b in prange(nb):
        for i in range(b * 64, min(n, (b + 1) * 64)):
            for j in range(F):
                v = V[i, j]
                if v < 0:
                    continue
                c = C[i, j]
                if c > cmax:
                    c = cmax
                if v == 1:
                    H1b[b, c] += 1
                else:
                    H0b[b, c] += 1
    return H1b.sum(axis=0), H0b.sum(axis=0)


@njit(parallel=True, cache=True)
def accumulate_entries(C, T1, explained_acc, predictive_acc):
    """explained_acc += (C >= 1); predictive_acc += P(x_ij = 1 | C_ij) for every entry."""
    n, F = C.shape
    rs = 1 if T1.shape[0] > 1 else 0
    for i in prange(n):
        for j in range(F):
            c = C[i, j]
            if c >= 1:
                explained_acc[i, j] += 1
            predictive_acc[i, j] += np.exp(T1[i * rs, c])


@njit(parallel=True, cache=True)
def row_histograms(V, C, cmax):
    """Per-sample count histograms: H1[i, c], H0[i, c] = observed present / absent entries of
    sample i with count c (capped at cmax)."""
    n, F = V.shape
    H1 = np.zeros((n, cmax + 1), np.int64)
    H0 = np.zeros((n, cmax + 1), np.int64)
    for i in prange(n):
        for j in range(F):
            v = V[i, j]
            if v < 0:
                continue
            c = C[i, j]
            if c > cmax:
                c = cmax
            if v == 1:
                H1[i, c] += 1
            else:
                H0[i, c] += 1
    return H1, H0
