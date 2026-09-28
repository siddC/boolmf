"""Synthetic binary matrices with known Boolean structure."""

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_boolean_factors", "make_nested_factors"]


def _observe(structure, detection, background, missing, rng):
    X = np.where(structure, rng.random(structure.shape) < detection,
                 rng.random(structure.shape) < background).astype(float)
    if missing > 0:
        X[rng.random(X.shape) < missing] = np.nan
    return X


def make_boolean_factors(n_samples=200, n_features=300, n_components=5, *, prevalence=(0.1, 0.5),
                         membership=(0.05, 0.2), detection=0.97, background=0.01, missing=0.0,
                         random_state=None, return_truth=False):
    """Random Boolean factors observed through noisy-OR style noise.

    Each component is active in each sample with a component-specific prevalence drawn
    uniformly from ``prevalence``, and contains each feature with a rate drawn from
    ``membership``. An entry is structurally present when an active component contains the
    feature; it is observed as present with probability ``detection`` if so, ``background``
    otherwise.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features), float with NaN for missing entries
    truth : dict, only if ``return_truth``
        ``members`` (n_components, n_features), ``activations`` (n_samples, n_components),
        ``structure`` (n_samples, n_features), all bool.
    """
    rng = check_random_state(random_state)
    pi = rng.uniform(*prevalence, size=n_components)
    rho = rng.uniform(*membership, size=n_components)
    A = rng.random((n_samples, n_components)) < pi
    M = rng.random((n_components, n_features)) < rho[:, None]
    S = (A.astype(np.int32) @ M.astype(np.int32)) > 0
    X = _observe(S, detection, background, missing, rng)
    if return_truth:
        return X, {"members": M, "activations": A, "structure": S}
    return X


def make_nested_factors(*, detection=0.97, background=0.005, n_noise_features=100,
                        missing=0.0, random_state=None, return_truth=False):
    """A demanding benchmark: exclusive groups plus overlapping, shared and nested components.

    500 samples fall into 8 exclusive groups (150 down to 15 samples), each with its own
    component of 80 features; groups 0 & 1 and 2 & 3 share 20 extra features. On top of
    them, 12 overlapping components span 5 to 40 features and 2% to 25% prevalence, including
    one enriched in a group, one restricted to 60% of a small group, one with a module present
    in only half of its carriers, and one nested inside another (all carriers of component 8
    also carry component 19's features). Three features are shared by three overlapping
    components and two by four. ``n_noise_features`` features appear at random in 0.5-2% of
    samples and belong to no component.

    Returns
    -------
    X : ndarray of shape (500, n_features)
    truth : dict, only if ``return_truth``
        ``members`` (20, n_features), ``activations`` (500, 20), ``structure`` and ``names``.
    """
    rng = check_random_state(random_state)
    sizes = [150, 100, 80, 60, 40, 30, 25, 15]
    n = sum(sizes)
    group = np.repeat(np.arange(8), sizes)
    rng.shuffle(group)
    G = 0

    def new(k):
        nonlocal G
        idx = list(range(G, G + k))
        G += k
        return idx

    members, acts, names = [], [], []
    group_feats = [new(80) for _ in range(8)]
    for a, b in [(0, 1), (2, 3)]:
        shared = new(20)
        group_feats[a] += shared
        group_feats[b] += shared
    for g in range(8):
        members.append(group_feats[g])
        acts.append(group == g)
        names.append(f"group {g} ({sizes[g]} samples)")
    specs = [(40, 0.25, "horizontal"), (40, 0.05, "horizontal"), (20, 0.10, "horizontal"),
             (20, 0.02, "horizontal"), (10, 0.10, "horizontal"), (10, 0.02, "horizontal"),
             (5, 0.10, "horizontal"), (5, 0.02, "horizontal"), (20, 0.10, "enriched"),
             (20, 0.60, "restricted"), (30, 0.15, "partial"), (15, 0.08, "nested")]
    ov_feats, ov_acts = [], []
    for size, prev, mode in specs:
        if mode == "enriched":
            ncar = int(prev * n)
            ing, outg = np.flatnonzero(group == 2), np.flatnonzero(group != 2)
            pick = np.concatenate([rng.choice(ing, int(0.8 * ncar), replace=False),
                                   rng.choice(outg, ncar - int(0.8 * ncar), replace=False)])
            c = np.zeros(n, bool)
            c[pick] = True
        elif mode == "restricted":
            c = (group == 5) & (rng.random(n) < prev)
        else:
            c = rng.random(n) < prev
        ov_feats.append(new(size))
        ov_acts.append(c)
    shared3, shared2 = new(3), new(2)
    for m in (0, 1, 2):
        ov_feats[m] += shared3
    for m in (0, 2, 4, 8):
        ov_feats[m] += shared2
    F_struct = G
    F = F_struct + n_noise_features

    S = np.zeros((n, F), bool)
    for fs, c in zip(members, acts):
        S[np.ix_(np.flatnonzero(c), fs)] = True
    module_carriers = None
    for m, (fs, c) in enumerate(zip(ov_feats, ov_acts)):
        fs = np.asarray(fs)
        if specs[m][2] == "partial":
            car = np.flatnonzero(c)
            S[np.ix_(car, fs[:20])] = True
            module_carriers = car[rng.random(car.size) < 0.5]
            S[np.ix_(module_carriers, fs[20:30])] = True
            S[np.ix_(car, fs[30:])] = True
        else:
            S[np.ix_(np.flatnonzero(c), fs)] = True
    S[np.ix_(np.flatnonzero(ov_acts[0]), ov_feats[11])] = True        # nesting
    for f in range(F_struct, F):
        S[rng.random(n) < rng.uniform(0.005, 0.02), f] = True

    X = _observe(S, detection, background, missing, rng)
    if not return_truth:
        return X
    all_members = members + [
        (fs[:20] if specs[m][2] == "partial" else fs) for m, fs in enumerate(ov_feats)]
    M = np.zeros((len(all_members), F), bool)
    for k, fs in enumerate(all_members):
        M[k, fs] = True
    A = np.stack(acts + ov_acts, axis=1)
    names += [f"overlap {m + 1:02d}: {s} features, {p:.0%} {mode}"
              for m, (s, p, mode) in enumerate(specs)]
    return X, {"members": M, "activations": A, "structure": S, "names": names,
               "n_groups": 8}
