"""Pinned components (``pin_init``) and entry-wise convergence of components.

* ``split_rhat`` from counts equals the split R-hat of the binary series themselves.
* Population moves never touch a pinned slot.
* A refit that pins the robust components of a first fit keeps each one in its own slot, in
  the order given, and its entries converge (R-hat near 1); fits without ``pin_init`` are
  unchanged.
"""

import numpy as np
import pytest

from boolmf import BayesianBooleanMF
from boolmf.consensus import robust_components, slot_components, split_rhat
from boolmf.datasets import make_boolean_factors
from boolmf.diagnostics import _rhat_basic, _split


def test_split_rhat_from_counts_matches_the_series():
    rng = np.random.default_rng(0)
    n_chains, n_draws, m = 3, 40, 25
    p = rng.uniform(0.05, 0.95, (n_chains, m))
    x = (rng.random((n_chains, n_draws, m)) < p[:, None, :]).astype(float)
    halves = x.reshape(n_chains, 2, n_draws // 2, m)
    ones = halves.sum(2).reshape(2 * n_chains, m)
    counts = np.full(2 * n_chains, n_draws // 2)
    got = split_rhat(ones, counts)
    for j in range(m):
        want = _rhat_basic(_split(x[:, :, j]))
        assert got[j] == pytest.approx(want, rel=1e-9)
    # constant and agreeing: 1; constant but disagreeing: inf; one usable split: NaN
    assert split_rhat(np.array([[0.0], [0.0]]), np.array([5, 5]))[0] == 1.0
    assert split_rhat(np.array([[0.0], [5.0]]), np.array([5, 5]))[0] == np.inf
    assert np.isnan(split_rhat(np.array([[2.0], [0.0]]), np.array([5, 1]))[0])


def _block_draws(n_chains=2, n_draws=20, seed=0):
    """Two planted blocks in fixed slots 0 and 1 (plus noise in slot 2), as (slots, Z, U)."""
    rng = np.random.default_rng(seed)
    n, F = 30, 40
    out = []
    for _c in range(n_chains):
        chain = []
        for _ in range(n_draws):
            Z = np.zeros((n, 3), bool)
            U = np.zeros((F, 3), bool)
            Z[:10, 0], U[:12, 0] = True, True
            Z[12:25, 1], U[20:30, 1] = True, True
            U[rng.integers(F), 0] ^= True
            Z[:, 2] = rng.random(n) < 0.2
            U[:, 2] = rng.random(F) < 0.2
            chain.append((np.array([3, 5, 9]), Z, U))
        out.append(chain)
    return out


def test_slot_components_follow_the_slots():
    rc = slot_components(_block_draws(), [5, 3], n_samples=30, n_features=40)
    assert rc.n_components == 2
    assert np.array_equal(rc.activations[:, 0] > 0.5, np.r_[[False] * 12, [True] * 13,
                                                            [False] * 5])
    assert rc.members[1, :12].min() > 0.8
    assert (rc.support == 1).all() and (rc.chain_support == 1).all()
    conv = rc.convergence()
    assert conv["activations"]["rhat_below"][1.01] == 1.0      # activations never change
    assert conv["members"]["n_entries"] >= 22


def test_robust_components_report_entry_rhat():
    draws = [[(Z, U) for _, Z, U in chain] for chain in _block_draws(3, 30, seed=1)]
    rc = robust_components(draws)
    assert rc.n_components == 2
    assert rc.member_rhat.shape == rc.members.shape
    assert rc.activation_rhat.shape == rc.activations.shape
    assert rc.chain_members.shape == (2, 3, 40)
    assert rc.chain_activations.shape == (30, 2, 3)
    conv = rc.convergence()
    assert conv["members"]["rhat_below"][1.1] > 0.9
    assert set(conv) == {"members", "activations", "components"}


def test_population_moves_leave_pinned_slots_alone():
    from types import SimpleNamespace

    from boolmf._model import loglik_tables
    from boolmf._sampler.chain import Member
    from boolmf._sampler.kernels import counts_from_state
    from boolmf._sampler.population import N_MOVES, crossover, transplant_or_delete

    rng = np.random.default_rng(4)
    V = (rng.random((8, 9)) < 0.4).astype(np.int8)
    K = 4
    movable = np.array([False, True, True, True])         # slot 0 pinned
    T1, T0 = (t[None, :] for t in loglik_tables("or_flip", 0.8, 0.2, K))
    moves = np.zeros((2, N_MOVES), np.int64)
    for _ in range(300):
        m = []
        for _c in range(2):
            Z = (rng.random((8, K)) < 0.4).astype(np.int8)
            U = (rng.random((9, K)) < 0.4).astype(np.int8)
            Z[:, 3] = U[:, 3] = 0
            eng = SimpleNamespace(C=counts_from_state(Z, U), T1=T1, T0=T0)
            m.append(Member(Z, U, eng, movable, (0.5, 1.0), (1.0, 1.0)))
        before = [(x.Z[:, 0].copy(), x.U[:, 0].copy()) for x in m]
        transplant_or_delete(V, m[0], m[1], rng, moves)
        crossover(V, m[0], m[1], "samples", np.arange(8), rng, moves)
        crossover(V, m[1], m[0], "features", np.arange(9), rng, moves)
        for x, (z, u) in zip(m, before):
            np.testing.assert_array_equal(x.Z[:, 0], z)
            np.testing.assert_array_equal(x.U[:, 0], u)
            np.testing.assert_array_equal(x.engine.C, counts_from_state(x.Z, x.U))
    assert moves[1].sum() > 0


FAST = dict(n_chains=3, max_sweeps=400, burn_in=200, n_draws=40, thin=5, random_state=0)


@pytest.fixture(scope="module")
def planted():
    X, truth = make_boolean_factors(150, 100, 4, prevalence=(0.2, 0.4), membership=(0.15, 0.3),
                                    detection=0.95, background=0.02, random_state=3,
                                    return_truth=True)
    first = BayesianBooleanMF(**FAST).fit(X)
    return X, truth, first.robust_components()


def test_pinned_refit_keeps_the_robust_components(planted):
    X, truth, rc = planted
    assert rc.n_components >= 4
    starts = [rc.sample(random_state=c) for c in range(3)]
    m = BayesianBooleanMF(pin_init=True, init=starts, max_components=rc.n_components + 10,
                          **FAST).fit(X)
    assert m.n_pinned_ == rc.n_components
    pc = m.pinned_components()
    assert pc.n_components == rc.n_components
    # the k-th pinned component is the k-th robust component, in every chain
    for k in range(rc.n_components):
        assert np.abs(pc.members[k] - rc.members[k]).max() < 0.35
        assert np.abs(pc.activations[:, k] - rc.activations[:, k]).max() < 0.35
    assert (pc.support > 0.95).all()
    conv = pc.convergence()
    assert conv["members"]["rhat_below"][1.1] > 0.95
    assert conv["activations"]["rhat_below"][1.1] > 0.95
    # the pinned components reconstruct the data about as well as the full posterior
    P = pc.reconstruct(m.detection_rate_, m.background_rate_)
    ll = np.mean(np.where(X == 1, np.log(P), np.log1p(-P)))
    assert ll > m.score(X, entries=np.ones(X.shape, bool)) - 0.02


def test_pin_init_checks():
    X = (np.random.default_rng(0).random((20, 15)) < 0.3).astype(float)
    with pytest.raises(ValueError, match="pin_init needs init"):
        BayesianBooleanMF(pin_init=True, n_chains=1, max_sweeps=10, burn_in=5,
                          n_draws=2).fit(X)
    init = (np.ones((5, 15), bool), np.ones((20, 5), bool))
    with pytest.raises(ValueError, match="do not fit"):
        BayesianBooleanMF(pin_init=True, init=init, max_components=3, n_chains=1,
                          max_sweeps=10, burn_in=5, n_draws=2).fit(X)
    two = [init, (np.ones((4, 15), bool), np.ones((20, 4), bool))]
    with pytest.raises(ValueError, match="same number"):
        BayesianBooleanMF(pin_init=True, init=two, n_chains=2, max_sweeps=10, burn_in=5,
                          n_draws=2).fit(X)
    with pytest.raises(ValueError, match="no pinned"):
        BayesianBooleanMF(n_chains=1, max_sweeps=10, burn_in=5, n_draws=2).fit(
            X).pinned_components()
    with pytest.warns(UserWarning, match="no free slot"):
        BayesianBooleanMF(pin_init=True, init=init, max_components=5, n_chains=1,
                          max_sweeps=10, burn_in=5, n_draws=2).fit(X)


def test_pin_init_runs_with_groups_rates_and_population(planted):
    X, _, rc = planted
    starts = [rc.sample(random_state=c) for c in range(3)]
    groups = np.arange(X.shape[1]) % 2
    m = BayesianBooleanMF(pin_init=True, init=starts, max_components=rc.n_components + 10,
                          feature_groups=groups, detection_effects=("sample",),
                          background_effects=("sample",), population_moves=5,
                          likelihood_power=0.8, n_chains=3, max_sweeps=60, burn_in=30,
                          n_draws=10, thin=1, random_state=0).fit(X)
    pc = m.pinned_components()
    assert pc.n_components == rc.n_components and (pc.support > 0.9).all()
