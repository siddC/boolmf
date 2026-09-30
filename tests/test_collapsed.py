"""The collapsed Indian buffet process sampler (``births="enumerate"`` / ``"metropolis"``).

On a problem small enough to enumerate (2 rows, 2 features, one of them missing in one row)
the exact posterior over the Boolean factorisation is a sum over multisets of component types
(which rows carry the component, which features it contains). With alpha = 1 and at most 8
components the truncation leaves out less than 1e-4 of the mass. The sampler's distribution of
the number of components and of the count matrix C (how many components cover each entry) must
match it. An earlier bug (a row's own components removed while its shared components were
still being updated) moved one probability of C by 0.015; sampling error here is below 0.005.
"""

import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from boolmf import BayesianBooleanMF, presets
from boolmf._model import loglik_tables
from boolmf._sampler.chain import _csr
from boolmf._sampler.collapsed import BIRTH_ENUMERATE, BIRTH_METROPOLIS, collapsed_rows
from boolmf._sampler.kernels import counts_from_state, update_memberships

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "papers"))

V = np.array([[1, 0], [1, -1]], np.int8)
N_ROWS, N_FEATURES = V.shape
ALPHA, P, KMAX = 1.0, 0.4, 8


def _exact(T1, T0):
    carriers = [(1, 0), (0, 1), (1, 1)]
    members = list(itertools.product([0, 1], repeat=N_FEATURES))
    types = [(h, u) for h in carriers for u in members]
    observed = V >= 0
    dist, kdist = {}, np.zeros(KMAX + 1)
    for K in range(KMAX + 1):
        for combo in itertools.combinations_with_replacement(range(len(types)), K):
            n = np.bincount(combo, minlength=len(types))
            lp = K * math.log(ALPHA) - sum(math.lgamma(c + 1) for c in n)
            C = np.zeros((N_ROWS, N_FEATURES), int)
            for t in combo:
                h, u = types[t]
                m = sum(h)
                lp += (math.lgamma(N_ROWS - m + 1) + math.lgamma(m) - math.lgamma(N_ROWS + 1)
                       + sum(u) * math.log(P) + (N_FEATURES - sum(u)) * math.log1p(-P))
                C += np.outer(h, u)
            lp += np.where(V == 1, T1[C], T0[C])[observed].sum()
            key = tuple(np.minimum(C, 3).ravel())
            dist[key] = dist.get(key, 0.0) + math.exp(lp)
            kdist[K] += math.exp(lp)
    z = kdist.sum()
    return {k: v / z for k, v in dist.items()}, kdist / z


@pytest.mark.parametrize("likelihood", ["noisy_or", "or_flip"])
@pytest.mark.parametrize("births", [BIRTH_ENUMERATE, BIRTH_METROPOLIS])
def test_collapsed_sampler_targets_exact_posterior(births, likelihood):
    T1, T0 = loglik_tables(likelihood, 0.8, 0.1, 30)
    exact, exact_k = _exact(T1, T0)
    slots = 24
    Z = np.zeros((N_ROWS, slots), np.int8)
    U = np.zeros((N_FEATURES, slots), np.int8)
    C = counts_from_state(Z, U)
    stats = np.zeros(3, np.int64)
    logit_p = np.full((1, slots), math.log(P / (1 - P)))
    rng = np.random.default_rng(0)
    n_iter = 20000
    hist, k_hist = {}, np.zeros(slots + 1)
    for _ in range(n_iter):
        collapsed_rows(V, Z, U, C, T1, T0, ALPHA, P, births, 10, False,
                       np.uint64(rng.integers(2**62)), stats)
        used = Z.sum(0) > 0
        U[:, ~used] = 0
        ptr, idx = _csr(Z)
        update_memberships(V, U, C, T1[None], T0[None], logit_p, ptr, idx, used,
                           np.uint64(rng.integers(2**62)))
        key = tuple(np.minimum(C, 3).ravel())
        hist[key] = hist.get(key, 0) + 1
        k_hist[used.sum()] += 1
    np.testing.assert_array_equal(C, counts_from_state(Z, U))
    assert stats[0] > 1000 and stats[2] == 0            # many births, never out of slots
    k_err = np.abs(k_hist[:KMAX + 1] / n_iter - exact_k)
    assert k_err.max() < 0.015
    top = sorted(exact, key=lambda k: -exact[k])[:8]
    c_err = [abs(hist.get(k, 0) / n_iter - exact[k]) for k in top]
    assert max(c_err) < 0.009


def test_birth_options_validation(small_data):
    X, _ = small_data
    with pytest.raises(ValueError, match="n_components=None"):
        BayesianBooleanMF(n_components=3, births="enumerate", split_merge=False).fit(X)
    with pytest.raises(ValueError, match="split_merge=False"):
        BayesianBooleanMF(births="enumerate", membership_level="shared").fit(X)
    with pytest.raises(ValueError, match="shared"):
        BayesianBooleanMF(births="enumerate", split_merge=False).fit(X)
    with pytest.raises(ValueError, match="births='enumerate'"):
        BayesianBooleanMF(births="metropolis", birth_members="gibbs", split_merge=False,
                          membership_level="shared").fit(X)
    with pytest.raises(ValueError):
        BayesianBooleanMF(birth_members="joint").fit(X)
    with pytest.raises(ValueError, match="n_components=None"):
        BayesianBooleanMF(n_components=3, ibp_side="features").fit(X)


@pytest.mark.parametrize("births, members", [("enumerate", "exact"), ("enumerate", "gibbs"),
                                             ("metropolis", "exact")])
def test_collapsed_sampler_recovers_simple_structure(births, members):
    rng = np.random.default_rng(0)
    Zt = rng.random((120, 3)) < 0.4
    Ut = np.zeros((40, 3), bool)
    for k in range(3):
        Ut[12 * k:12 * k + 12, k] = True
    truth = (Zt.astype(int) @ Ut.T.astype(int)) > 0
    X = np.where(rng.random(truth.shape) < 0.03, ~truth, truth).astype(float)
    burn_in = 150 if births == "enumerate" else 2000              # MH births are slow
    m = BayesianBooleanMF(births=births, birth_members=members, membership_level="shared",
                          split_merge=False, init="empty", n_chains=2, n_draws=50, thin=1,
                          random_state=0, burn_in=burn_in).fit(X)
    assert m.n_components_ == 3
    found = m.components_ > 0.5
    for k in range(3):
        assert any((found[j] == Ut[:, k]).all() for j in range(found.shape[0]))


@pytest.mark.parametrize("name, n, bound", [("degree1", 100, 0.8), ("disconnected", 100, 3.0)])
def test_reproduces_wood2006_structures(name, n, bound):
    from wood2006 import run

    ind, struct, _, _ = run(name, n_datasets=4, iterations=(n,))[n]
    assert ind < bound and struct < bound


def test_ibp_on_features_transposes_results():
    from wood2006 import STRUCTURES, simulate

    X = simulate(STRUCTURES["disconnected"], np.random.default_rng(0))     # trials x observed
    params = {**presets.wood2006, "detection_prior": 0.9, "background_prior": 0.01,
              "activation_prior": 0.1, "alpha_prior": 3.0, "burn_in": 200, "n_draws": 100,
              "thin": 1, "min_support": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BayesianBooleanMF(random_state=0, **params).fit(X)
    assert m.components_.shape == (m.n_components_, X.shape[1])
    assert m.activations_.shape == (X.shape[0], m.n_components_)
    children = {tuple(np.flatnonzero(c > 0.5)) for c in m.components_}
    assert {(0, 1, 2), (2,), (3, 4), (3, 5)} <= children


def test_rukat_yau2019_preset_finds_rank_without_noise():
    from rukat_yau2019 import posterior_mode

    assert posterior_mode(8, 0.0, seed=0) == 8
