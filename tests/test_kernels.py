"""The numba kernels must match a plain-Python implementation draw for draw."""

import numpy as np

from boolmf._model import loglik_tables
from boolmf._sampler.chain import _csr
from boolmf._sampler.kernels import counts_from_state, update_activations, update_memberships

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def _next(state):
    state = (state + GOLDEN) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    z ^= z >> 31
    return state, (z >> 11) * (1.0 / 9007199254740992.0)


def _row_state(seed, row):
    s = (seed ^ (((row + 1) * GOLDEN) & M64)) & M64
    s, _ = _next(s)
    return s


def _ref_update_activations(V, Z, C, T1, T0, logit_pi, seed):
    n, K = Z.shape
    for i in range(n):
        state = _row_state(seed, i)
        for k in range(K):
            members = np.flatnonzero(U_REF[:, k])
            lo = logit_pi[k]
            old = Z[i, k]
            for j in members:
                v = V[i, j]
                if v < 0:
                    continue
                cm = C[i, j] - old
                lo += (T1[cm + 1] - T1[cm]) if v == 1 else (T0[cm + 1] - T0[cm])
            state, u = _next(state)
            new = 1 if u * (1.0 + np.exp(-lo)) < 1.0 else 0
            if new != old:
                C[i, members] += new - old
                Z[i, k] = new


U_REF = None


def test_activation_kernel_matches_reference():
    global U_REF
    rng = np.random.default_rng(0)
    n, F, K = 30, 25, 5
    V = (rng.random((n, F)) < 0.3).astype(np.int8)
    V[rng.random((n, F)) < 0.1] = -1
    Z = (rng.random((n, K)) < 0.4).astype(np.int8)
    U = (rng.random((F, K)) < 0.3).astype(np.int8)
    U_REF = U
    C = counts_from_state(Z, U)
    T1, T0 = loglik_tables("noisy_or", 0.9, 0.05, K)
    logit_pi = rng.normal(size=K)
    seed = 123456789
    Z1, C1 = Z.copy(), C.copy()
    ptr, idx = _csr(U)
    update_activations(V, Z1, C1, T1, T0, logit_pi, ptr, idx, np.ones(K, bool), np.uint64(seed))
    Z2, C2 = Z.copy(), C.copy().astype(np.int64)
    _ref_update_activations(V, Z2, C2, T1, T0, logit_pi, seed)
    np.testing.assert_array_equal(Z1, Z2)
    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(C1, counts_from_state(Z1, U))


def test_membership_kernel_keeps_counts_consistent():
    rng = np.random.default_rng(1)
    n, F, K = 40, 30, 6
    V = (rng.random((n, F)) < 0.3).astype(np.int8)
    Z = (rng.random((n, K)) < 0.4).astype(np.int8)
    U = (rng.random((F, K)) < 0.3).astype(np.int8)
    C = counts_from_state(Z, U)
    T1, T0 = loglik_tables("or_flip", 0.9, 0.05, K)
    ptr, idx = _csr(Z)
    update_memberships(V, U, C, T1, T0, rng.normal(size=K), ptr, idx, np.ones(K, bool),
                       np.uint64(7))
    np.testing.assert_array_equal(C, counts_from_state(Z, U))
