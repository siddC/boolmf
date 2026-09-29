"""Split–merge moves must leave the exact posterior invariant.

On a problem small enough to enumerate (3 samples, 3 features, an anchor slot and two free
slots, one missing entry), draw states from the exact posterior, apply split–merge moves, and
check that the states still follow the exact posterior. The number of used slots is the most
sensitive statistic: a wrong move-selection probability shifts it clearly.
"""

import numpy as np
import pytest
from scipy import stats

from boolmf._model import loglik_tables
from boolmf._sampler.kernels import counts_from_state
from boolmf._sampler.splitmerge import _log_beta_bernoulli, split_merge_moves

V = np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], np.int8)
N_SAMPLES, N_FEATURES = V.shape
ANCHOR = np.array([0, 0, 1], np.int8)          # slot 0: active everywhere, fixed members
FREE = np.array([False, True, True])
ZPRIOR, UPRIOR = (0.3, 1.0), (1.0, 2.0)
BITS_Z, BITS_U = N_SAMPLES * 2, N_FEATURES * 2


def _state(code):
    zbits, ubits = code >> BITS_U, code & ((1 << BITS_U) - 1)
    Z = np.ones((N_SAMPLES, 3), np.int8)
    U = np.zeros((N_FEATURES, 3), np.int8)
    U[:, 0] = ANCHOR
    Z[:, 1:] = np.array([(zbits >> t) & 1 for t in range(BITS_Z)]).reshape(N_SAMPLES, 2)
    U[:, 1:] = np.array([(ubits >> t) & 1 for t in range(BITS_U)]).reshape(N_FEATURES, 2)
    return Z, U


def _code(Z, U):
    zbits = sum(int(v) << t for t, v in enumerate(Z[:, 1:].ravel()))
    ubits = sum(int(v) << t for t, v in enumerate(U[:, 1:].ravel()))
    return (zbits << BITS_U) | ubits


def _exact_posterior(T1, T0):
    logp = np.empty(1 << (BITS_Z + BITS_U))
    observed = V >= 0
    for code in range(logp.size):
        Z, U = _state(code)
        C = counts_from_state(Z, U).astype(int)
        lp = np.where(V == 1, T1[C], T0[C])[observed].sum()
        for k in (1, 2):
            lp += _log_beta_bernoulli(int(Z[:, k].sum()), *ZPRIOR, N_SAMPLES)
            lp += _log_beta_bernoulli(int(U[:, k].sum()), *UPRIOR, N_FEATURES)
        logp[code] = lp
    p = np.exp(logp - logp.max())
    return p / p.sum()


@pytest.mark.parametrize("likelihood", ["or_flip", "noisy_or"])
def test_split_merge_preserves_exact_posterior(likelihood):
    T1, T0 = loglik_tables(likelihood, 0.8, 0.2, 3)
    p = _exact_posterior(T1, T0)
    rng = np.random.default_rng(1)
    n_draws = 20000
    counts = np.zeros(p.size)
    moves = np.zeros((2, 5), np.int64)
    for code in rng.choice(p.size, n_draws, p=p):
        Z, U = _state(code)
        C = counts_from_state(Z, U)
        split_merge_moves(V, Z, U, C, T1, T0, FREE, ZPRIOR, UPRIOR, 2, 2, rng, moves)
        np.testing.assert_array_equal(C, counts_from_state(Z, U))
        assert (Z[:, 0] == 1).all() and (U[:, 0] == ANCHOR).all()      # anchor untouched
        counts[_code(Z, U)] += 1
    assert (moves[1] > 40).all()                     # every move type is accepted often

    expected = p * n_draws
    used = np.array([sum(int(Z[:, k].any() and U[:, k].any()) for k in (1, 2))
                     for Z, U in map(_state, range(p.size))])
    obs_used = np.array([counts[used == c].sum() for c in range(3)])
    exp_used = np.array([expected[used == c].sum() for c in range(3)])
    assert stats.chi2.sf(((obs_used - exp_used) ** 2 / exp_used).sum(), 2) > 1e-3
    big = expected >= 5
    chi2 = ((counts[big] - expected[big]) ** 2 / expected[big]).sum()
    assert stats.chi2.sf(chi2, big.sum() - 1) > 1e-3


def test_split_merge_with_three_free_slots():
    """With three free slots the partner of a move is drawn by similarity, so the reverse
    selection probability differs from the forward one. Six moves per draw make a wrong
    acceptance ratio visible in the full state distribution as well as in the used-slot count."""
    V3 = np.array([[1, 1], [1, 0], [0, 1]], np.int8)
    n, F, K = 3, 2, 3
    zprior, uprior = (0.4, 1.0), (1.0, 1.0)
    T1, T0 = loglik_tables("or_flip", 0.8, 0.2, K)
    bz, bu = n * K, F * K
    codes = np.arange(1 << (bz + bu))
    Zs = (((codes[:, None] >> bu) >> np.arange(bz)) & 1).reshape(-1, n, K).astype(np.int8)
    Us = ((codes[:, None] >> np.arange(bu)) & 1).reshape(-1, F, K).astype(np.int8)
    C = np.einsum("sik,sjk->sij", Zs.astype(int), Us.astype(int))
    logp = np.where(V3[None] == 1, T1[C], T0[C]).sum((1, 2))
    for s in range(codes.size):
        for k in range(K):
            logp[s] += _log_beta_bernoulli(int(Zs[s, :, k].sum()), *zprior, n)
            logp[s] += _log_beta_bernoulli(int(Us[s, :, k].sum()), *uprior, F)
    p = np.exp(logp - logp.max())
    p /= p.sum()
    used = ((Zs.sum(1) > 0) & (Us.sum(1) > 0)).sum(1)
    wz, wu = 1 << np.arange(bz), 1 << np.arange(bu)

    rng = np.random.default_rng(11)
    n_draws = 100000
    moves = np.zeros((2, 5), np.int64)
    counts = np.zeros(p.size)
    for code in rng.choice(p.size, n_draws, p=p):
        Z, U = Zs[code].copy(), Us[code].copy()
        Cs = counts_from_state(Z, U)
        split_merge_moves(V3, Z, U, Cs, T1, T0, np.ones(K, bool), zprior, uprior, 6, 2, rng,
                          moves)
        counts[(int(Z.ravel() @ wz) << bu) | int(U.ravel() @ wu)] += 1
    assert (moves[1] > 1000).all()
    expected = p * n_draws
    obs_used = np.array([counts[used == c].sum() for c in range(K + 1)])
    exp_used = np.array([expected[used == c].sum() for c in range(K + 1)])
    assert stats.chi2.sf(((obs_used - exp_used) ** 2 / exp_used).sum(), K) > 1e-3
    big = expected >= 5
    chi2 = ((counts[big] - expected[big]) ** 2 / expected[big]).sum()
    assert stats.chi2.sf(chi2, big.sum() - 1) > 1e-3
