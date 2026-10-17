"""Kernels for ``noisy_or`` with per-component detection rates (the log-survival form).

Each active component k containing feature j delivers it in sample i independently with
probability lambda_ik, and the background with probability b_i, so

    log P(x_ij = 0) = LB0[i] + L[i, j],   L[i, j] = sum over covering k of S[i, k],

with S[i, k] = log(1 - lambda_ik) and LB0[i] = log(1 - b_i). ``logit lambda_ik = off_i + h_k``:
h_k is the component's logit rate and off_i the sample's offset (0 without per-sample rates).
S, LB0 and off have one row per sample or a single shared row (row index i * rs).

Rates are updated one component at a time by slice sampling (Neal 2003) on the exact
likelihood of the entries the component covers; sample offsets and per-sample background rates
row by row. Randomness uses the counter-based streams of ``kernels``.
"""

import math

import numpy as np
from numba import njit, prange

from .kernels import _decide, _next_uniform, _row_state

LOG_EPS = math.log(1e-12)


@njit(inline="always")
def entry_ll(v, lb0, L):
    """log P(x = v) given log P(x = 0) = lb0 + L."""
    x = lb0 + L
    if v == 1:
        return math.log(-math.expm1(x)) if x < -1e-12 else LOG_EPS
    return x


@njit(inline="always")
def log1m_sigmoid(x):
    """log(1 - sigmoid(x)) = -softplus(x), computed stably."""
    if x > 0:
        return -x - math.log1p(math.exp(-x))
    return -math.log1p(math.exp(x))


@njit(parallel=True, cache=True)
def survival_matrix(off, h):
    """S[i, k] = log(1 - sigmoid(off[i] + h[k])), one row per entry of ``off``."""
    S = np.empty((off.size, h.size))
    for i in prange(off.size):
        for k in range(h.size):
            S[i, k] = log1m_sigmoid(off[i] + h[k])
    return S


@njit(parallel=True, cache=True)
def logsurv_from_state(Z, S, mem_ptr, mem_idx, F):
    """L[i, j] = sum over components k active in i and containing j of S[i, k]."""
    n, K = Z.shape
    rs = 1 if S.shape[0] > 1 else 0
    L = np.zeros((n, F))
    for i in prange(n):
        for k in range(K):
            if Z[i, k] == 0:
                continue
            s = S[i * rs, k]
            for t in range(mem_ptr[k], mem_ptr[k + 1]):
                L[i, mem_idx[t]] += s
    return L


@njit(parallel=True, cache=True)
def update_memberships_ls(V, U, L, LB0, S, logit_rho, act_ptr, act_idx, update_mask, seed,
                          metropolis=False):
    """Gibbs update of U[j, k] for every feature j (in parallel) and component k."""
    F, K = U.shape
    rs = 1 if S.shape[0] > 1 else 0
    rb = 1 if LB0.shape[0] > 1 else 0
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
                s = S[i * rs, k]
                Lm = L[i, j] - old * s
                lb = LB0[i * rb]
                lo += entry_ll(v, lb, Lm + s) - entry_ll(v, lb, Lm)
            state, new = _decide(state, lo, old, metropolis)
            if new != old:
                d = new - old
                for t in range(act_ptr[k], act_ptr[k + 1]):
                    i = act_idx[t]
                    L[i, j] += d * S[i * rs, k]
                U[j, k] = new


@njit(parallel=True, cache=True)
def update_activations_ls(V, Z, L, LB0, S, logit_pi, mem_ptr, mem_idx, update_mask, seed,
                          metropolis=False):
    """Gibbs update of Z[i, k] for every sample i (in parallel) and component k."""
    n, K = Z.shape
    rs = 1 if S.shape[0] > 1 else 0
    rb = 1 if LB0.shape[0] > 1 else 0
    pz = 1 if logit_pi.shape[0] > 1 else 0
    for i in prange(n):
        state = _row_state(seed, i)
        lb = LB0[i * rb]
        for k in range(K):
            if not update_mask[k]:
                continue
            lo = logit_pi[i * pz, k]
            old = Z[i, k]
            s = S[i * rs, k]
            for t in range(mem_ptr[k], mem_ptr[k + 1]):
                j = mem_idx[t]
                v = V[i, j]
                if v < 0:
                    continue
                Lm = L[i, j] - old * s
                lo += entry_ll(v, lb, Lm + s) - entry_ll(v, lb, Lm)
            state, new = _decide(state, lo, old, metropolis)
            if new != old:
                d = new - old
                for t in range(mem_ptr[k], mem_ptr[k + 1]):
                    L[i, mem_idx[t]] += d * s
                Z[i, k] = new


@njit(parallel=True, cache=True)
def accumulate_entries_ls(C, L, LB0, explained_acc, predictive_acc):
    """explained_acc += (C >= 1); predictive_acc += P(x_ij = 1) for every entry."""
    n, F = L.shape
    rb = 1 if LB0.shape[0] > 1 else 0
    for i in prange(n):
        lb = LB0[i * rb]
        for j in range(F):
            if C[i, j] >= 1:
                explained_acc[i, j] += 1
            predictive_acc[i, j] += -math.expm1(lb + L[i, j])


@njit(cache=True)
def _row_total(V, L, i, lb):
    acc = 0.0
    for j in range(V.shape[1]):
        v = V[i, j]
        if v >= 0:
            acc += entry_ll(v, lb, L[i, j])
    return acc


@njit(parallel=True, cache=True)
def total_loglik_ls(V, L, LB0):
    n = V.shape[0]
    rb = 1 if LB0.shape[0] > 1 else 0
    rows = np.zeros(n)
    for i in prange(n):
        rows[i] = _row_total(V, L, i, LB0[i * rb])
    return rows.sum()


@njit(parallel=True, cache=True)
def project_activations_ls(V, U, s_row, lb0, logit_pi, mem_ptr, mem_idx, row_seeds, n_sweeps,
                           n_keep):
    """Activations of new samples with memberships and rates held fixed (see
    ``kernels.project_activations``); s_row[k] = log(1 - lambda_k) at the population level."""
    n, F = V.shape
    K = U.shape[1]
    out = np.zeros((n, K))
    for i in prange(n):
        state = np.uint64(row_seeds[i])
        z = np.zeros(K, np.int8)
        Li = np.zeros(F)
        for sw in range(n_sweeps):
            for k in range(K):
                lo = logit_pi[k]
                old = z[k]
                s = s_row[k]
                for t in range(mem_ptr[k], mem_ptr[k + 1]):
                    j = mem_idx[t]
                    v = V[i, j]
                    if v < 0:
                        continue
                    Lm = Li[j] - old * s
                    lo += entry_ll(v, lb0, Lm + s) - entry_ll(v, lb0, Lm)
                state, u = _next_uniform(state)
                new = np.int8(1) if u * (1.0 + np.exp(-lo)) < 1.0 else np.int8(0)
                if new != old:
                    d = new - old
                    for t in range(mem_ptr[k], mem_ptr[k + 1]):
                        Li[mem_idx[t]] += d * s
                    z[k] = new
            if sw >= n_sweeps - n_keep:
                for k in range(K):
                    out[i, k] += z[k]
        for k in range(K):
            out[i, k] /= n_keep
    return out


# --------------------------------------------------------------------------- rate updates
@njit(parallel=True, cache=True)
def _component_loglik(V, L, LB0, S, off, hv, k, act_ptr, act_idx, mem_ptr, mem_idx):
    """Log-likelihood of the entries covered by component k if its logit rate were hv."""
    rs = 1 if S.shape[0] > 1 else 0
    ro = 1 if off.shape[0] > 1 else 0
    rb = 1 if LB0.shape[0] > 1 else 0
    a0 = act_ptr[k]
    nc = act_ptr[k + 1] - a0
    rows = np.zeros(nc)
    for t in prange(nc):
        i = act_idx[a0 + t]
        rows[t] = _carrier_ll(V, L, i, S[i * rs, k], log1m_sigmoid(off[i * ro] + hv),
                              LB0[i * rb], k, mem_ptr, mem_idx)
    return rows.sum()


@njit(cache=True)
def _carrier_ll(V, L, i, s_old, s_new, lb, k, mem_ptr, mem_idx):
    acc = 0.0
    for q in range(mem_ptr[k], mem_ptr[k + 1]):
        j = mem_idx[q]
        v = V[i, j]
        if v >= 0:
            acc += entry_ll(v, lb, L[i, j] - s_old + s_new)
    return acc


@njit(cache=True)
def _slice_step(state, x0, f0, lo_bound, hi_bound):
    """Slice-sampling helper: returns (state, level, left, right) for a unit-width bracket
    around x0 (no stepping out; the caller steps out)."""
    state, u = _next_uniform(state)
    level = f0 + math.log(max(u, 1e-300))
    state, u = _next_uniform(state)
    left = max(x0 - u, lo_bound)
    right = min(left + 1.0, hi_bound)
    return state, level, left, right


@njit(cache=True)
def _component_target(V, L, LB0, S, off, x, k, act_ptr, act_idx, mem_ptr, mem_idx, mean_h, tau):
    d = (x - mean_h) / tau
    return _component_loglik(V, L, LB0, S, off, x, k, act_ptr, act_idx, mem_ptr, mem_idx) \
        - 0.5 * d * d


@njit(cache=True)
def update_component_rates(V, Z, U, L, LB0, S, off, h, mean_h, tau, act_ptr, act_idx,
                           mem_ptr, mem_idx, used, seed):
    """Slice-sample h[k] for every used component, one at a time (each update conditions on
    the others through L), keeping S and L in sync. Unused components draw h from the prior."""
    rs = 1 if S.shape[0] > 1 else 0
    ro = 1 if off.shape[0] > 1 else 0
    state = _row_state(seed, 0)
    for k in range(h.size):
        if not used[k]:
            state, u1 = _next_uniform(state)
            state, u2 = _next_uniform(state)
            h[k] = mean_h + tau * math.sqrt(-2.0 * math.log(max(u1, 1e-300))) \
                * math.cos(2.0 * math.pi * u2)
            for r in range(S.shape[0]):
                S[r, k] = log1m_sigmoid(off[r * ro] + h[k])
            continue
        h0 = h[k]
        f0 = _component_target(V, L, LB0, S, off, h0, k, act_ptr, act_idx, mem_ptr, mem_idx,
                               mean_h, tau)
        state, level, left, right = _slice_step(state, h0, f0, -np.inf, np.inf)
        steps = 32
        while steps > 0 and _component_target(V, L, LB0, S, off, left, k, act_ptr, act_idx,
                                              mem_ptr, mem_idx, mean_h, tau) > level:
            left -= 1.0
            steps -= 1
        steps = 32
        while steps > 0 and _component_target(V, L, LB0, S, off, right, k, act_ptr, act_idx,
                                              mem_ptr, mem_idx, mean_h, tau) > level:
            right += 1.0
            steps -= 1
        h1 = h0
        for _ in range(200):
            state, u = _next_uniform(state)
            x = left + (right - left) * u
            if _component_target(V, L, LB0, S, off, x, k, act_ptr, act_idx, mem_ptr, mem_idx,
                                 mean_h, tau) > level:
                h1 = x
                break
            if x < h0:
                left = x
            else:
                right = x
        if h1 != h0:
            h[k] = h1
            for t in range(act_ptr[k], act_ptr[k + 1]):
                i = act_idx[t]
                s_old = S[i * rs, k]
                s_new = log1m_sigmoid(off[i * ro] + h1)
                for q in range(mem_ptr[k], mem_ptr[k + 1]):
                    L[i, mem_idx[q]] += s_new - s_old
            for r in range(S.shape[0]):
                S[r, k] = log1m_sigmoid(off[r * ro] + h1)


@njit(cache=True)
def _row_detection_ll(V, i, y, h, lb, Zi, mem_ptr, mem_idx, acc, mark, touched):
    """Log-likelihood of the covered entries of row i if its detection offset were y."""
    nt = 0
    for k in range(h.size):
        if Zi[k] == 0:
            continue
        s = log1m_sigmoid(y + h[k])
        for t in range(mem_ptr[k], mem_ptr[k + 1]):
            j = mem_idx[t]
            if mark[j] == 0:
                mark[j] = 1
                touched[nt] = j
                nt += 1
            acc[j] += s
    tot = 0.0
    for q in range(nt):
        j = touched[q]
        v = V[i, j]
        if v >= 0:
            tot += entry_ll(v, lb, acc[j])
        acc[j] = 0.0
        mark[j] = 0
    return tot


@njit(cache=True)
def _offset_target(V, i, y, h, lb, Zi, mem_ptr, mem_idx, acc, mark, touched, mu, sigma):
    d = (y - mu) / sigma
    return _row_detection_ll(V, i, y, h, lb, Zi, mem_ptr, mem_idx, acc, mark, touched) \
        - 0.5 * d * d


@njit(parallel=True, cache=True)
def update_sample_offsets(V, Z, L, LB0, S, y, h, mu, sigma, mem_ptr, mem_idx, seed):
    """Slice-sample each sample's detection offset y[i] (logit lambda_ik = y_i + h_k), rows in
    parallel; then refresh S and L for that row."""
    n, F = V.shape
    rb = 1 if LB0.shape[0] > 1 else 0
    for i in prange(n):
        state = _row_state(seed, i)
        acc = np.zeros(F)
        mark = np.zeros(F, np.int8)
        touched = np.empty(F, np.int64)
        lb = LB0[i * rb]
        Zi = Z[i]
        y0 = y[i]
        f0 = _offset_target(V, i, y0, h, lb, Zi, mem_ptr, mem_idx, acc, mark, touched, mu,
                            sigma)
        state, level, left, right = _slice_step(state, y0, f0, -np.inf, np.inf)
        steps = 32
        while steps > 0 and _offset_target(V, i, left, h, lb, Zi, mem_ptr, mem_idx, acc, mark,
                                           touched, mu, sigma) > level:
            left -= 1.0
            steps -= 1
        steps = 32
        while steps > 0 and _offset_target(V, i, right, h, lb, Zi, mem_ptr, mem_idx, acc, mark,
                                           touched, mu, sigma) > level:
            right += 1.0
            steps -= 1
        y1 = y0
        for _ in range(200):
            state, u = _next_uniform(state)
            x = left + (right - left) * u
            if _offset_target(V, i, x, h, lb, Zi, mem_ptr, mem_idx, acc, mark, touched, mu,
                              sigma) > level:
                y1 = x
                break
            if x < y0:
                left = x
            else:
                right = x
        y[i] = y1
        for k in range(h.size):
            S[i, k] = log1m_sigmoid(y1 + h[k])
        for j in range(F):
            L[i, j] = 0.0
        for k in range(h.size):
            if Zi[k] == 0:
                continue
            for t in range(mem_ptr[k], mem_ptr[k + 1]):
                L[i, mem_idx[t]] += S[i, k]


@njit(cache=True)
def _background_target(V, L, i, y, mu, sigma):
    lb = log1m_sigmoid(y)
    tot = 0.0
    for j in range(V.shape[1]):
        v = V[i, j]
        if v >= 0:
            tot += entry_ll(v, lb, L[i, j])
    d = (y - mu) / sigma
    return tot - 0.5 * d * d


@njit(parallel=True, cache=True)
def update_sample_backgrounds(V, L, yb, mu, sigma, seed):
    """Slice-sample each sample's logit background rate yb[i], rows in parallel."""
    n = V.shape[0]
    for i in prange(n):
        state = _row_state(seed, i)
        y0 = yb[i]
        f0 = _background_target(V, L, i, y0, mu, sigma)
        state, level, left, right = _slice_step(state, y0, f0, -np.inf, np.inf)
        steps = 32
        while steps > 0 and _background_target(V, L, i, left, mu, sigma) > level:
            left -= 1.0
            steps -= 1
        steps = 32
        while steps > 0 and _background_target(V, L, i, right, mu, sigma) > level:
            right += 1.0
            steps -= 1
        for _ in range(200):
            state, u = _next_uniform(state)
            x = left + (right - left) * u
            if _background_target(V, L, i, x, mu, sigma) > level:
                yb[i] = x
                break
            if x < y0:
                left = x
            else:
                right = x
