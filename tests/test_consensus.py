"""Robust components from all draws, and the window-stability check."""

import numpy as np
import pytest

from boolmf.consensus import robust_components, robust_stability

N, F = 40, 60


def _block(rows, cols):
    z = np.zeros(N, bool)
    u = np.zeros(F, bool)
    z[rows] = True
    u[cols] = True
    return z, u


PLANTED = [_block(slice(0, 10), slice(0, 15)), _block(slice(15, 30), slice(20, 28))]


def _draws(n_chains=3, n_draws=40, seed=0, late=None, only_chain0=None):
    """Planted components in every draw (one extra or missing member now and then), plus
    random transient components that never repeat. ``late``: a component present only in the
    second half of every chain. ``only_chain0``: a component present only in chain 0."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_chains):
        chain = []
        for t in range(n_draws):
            zs, us = [], []
            for z, u in PLANTED:
                z, u = z.copy(), u.copy()
                u[rng.integers(F)] ^= rng.random() < 0.3
                zs.append(z)
                us.append(u)
            if late is not None and t >= n_draws // 2:
                zs.append(late[0])
                us.append(late[1])
            if only_chain0 is not None and c == 0:
                zs.append(only_chain0[0])
                us.append(only_chain0[1])
            for _ in range(3):                    # transient: random small rectangles
                zs.append(rng.random(N) < 0.1)
                us.append(rng.random(F) < 0.1)
            chain.append((np.stack(zs, 1), np.stack(us, 1)))
        out.append(chain)
    return out


def test_planted_components_are_found_and_transients_are_not():
    rc = robust_components(_draws())
    assert rc.n_components == 2
    for z, u in PLANTED:
        k = int(np.argmax(rc.members @ u))
        assert rc.support[k] == 1.0
        assert (rc.chain_support[k] == 1.0).all()
        assert rc.members[k, u].min() > 0.9 and rc.members[k, ~u].max() < 0.1
        assert np.array_equal(rc.activations[:, k] > 0.5, z)


def test_min_chains_and_min_draw_share():
    extra = _block(slice(32, 38), slice(40, 50))
    draws = _draws(n_chains=2, only_chain0=extra)       # in half of all draws, one chain
    assert robust_components(draws).n_components == 2
    assert robust_components(draws, min_chains=1).n_components == 3
    assert robust_components(draws, min_chains=1, min_draw_share=0.6).n_components == 2


def test_coverage_and_reconstruct():
    rc = robust_components(_draws())
    cov = rc.coverage()
    assert cov.shape == (N, F) and cov.min() >= 0 and cov.max() <= 1
    z, u = PLANTED[0]
    assert cov[np.ix_(z, u)].min() > 0.9
    planted = np.zeros((N, F), bool)
    for zz, uu in PLANTED:
        planted |= np.outer(zz, uu)
    assert cov[~planted].max() < 0.1
    P = rc.reconstruct(0.9, np.full(N, 0.05))
    np.testing.assert_allclose(P, 0.05 + 0.85 * cov)


def test_stability_flags_a_component_that_appears_late():
    stable = robust_stability(_draws(), n_windows=4)
    assert [w["share_of_last"] for w in stable] == [1.0, 1.0, 1.0, 1.0]
    late = robust_stability(_draws(late=_block(slice(30, 36), slice(50, 58))), n_windows=4)
    assert [w["n_robust"] for w in late] == [2, 2, 3, 3]
    assert late[0]["found_in_last"] == 2 and late[0]["share_of_last"] == pytest.approx(2 / 3)
    assert late[0]["kept_in_last"] == 2


def test_validation():
    with pytest.raises(ValueError, match="at least one draw"):
        robust_components([[]])
    with pytest.raises(ValueError, match="n_windows"):
        robust_stability(_draws(n_draws=3), n_windows=4)


def test_estimator_methods(fitted, small_data):
    _, truth = small_data
    rc = fitted.robust_components()
    assert rc.n_components == 3
    J = []
    for u in truth["members"].astype(bool):
        m = rc.members >= 0.5
        J.append(max((m[k] & u).sum() / (m[k] | u).sum() for k in range(rc.n_components)))
    assert min(J) > 0.8
    st = fitted.robust_stability(n_windows=2)
    assert len(st) == 2 and st[-1]["share_of_last"] == 1.0
