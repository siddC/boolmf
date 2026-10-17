"""Population moves: chains that exchange components (evolutionary Monte Carlo).

Independent chains of a Boolean factorization often settle in different modes: each finds part
of the structure, and none can reach the others' components by one-at-a-time updates or
split-merge moves. Here the chains run in lockstep and, every few sweeps, propose moves that
use the other chains' components, in the spirit of evolutionary Monte Carlo (Liang & Wong
2000, Statistica Sinica 10: 317-342) and its population moves for an unknown number of
components (Jasra, Stephens & Holmes 2007, Biometrika 94: 787-807).

The population targets the product of the chains' posteriors. A Metropolis-Hastings move on
chain i whose proposal depends on another chain's current state, held fixed during the move,
leaves chain i's posterior invariant for every value of that state, so it leaves the product
invariant (the argument of differential-evolution MCMC, ter Braak 2006). Each chain still
samples its own posterior; the others only supply proposals.

Moves, on the free slots; like split-merge moves they target the posterior with the
component probabilities pi and rho integrated out (Beta-Bernoulli), given each chain's own
rates and alpha, and each chain redraws pi and rho right after:

* transplant: copy a component (carriers and members) of a donor chain into an empty slot.
* delete: empty a slot whose component is identical to one of the donor's (the reverse of a
  transplant).
* crossover: swap between two chains every component that lies inside a region of the data,
  either a group of samples (components whose carriers all lie in it) or a group of features
  (members all in it). Regions are the clusters of a fixed hierarchical clustering of the data
  (average linkage, Jaccard distance), so they never depend on the chains' states and choosing
  one is symmetric. Incoming components take a uniformly random set of the receiving chain's
  freed and empty slots.
"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.special import gammaln

from .chain import _config_key, chain_steps

TRANSPLANT, DELETE, CROSS_SAMPLES, CROSS_FEATURES = 0, 1, 2, 3
MOVE_NAMES = ("transplant", "delete", "crossover_samples", "crossover_features")
N_MOVES = 4
_FORMAT = 1


@dataclass
class PopulationConfig:
    every: int = 10                   # sweeps between rounds of population moves
    n_transplant: int = 20            # transplant/delete proposals per chain and round
    n_crossover: int = 10             # crossover proposals per round
    region_min_size: int = 3          # smallest region (samples or features)
    region_max_fraction: float = 0.5  # largest region, as a fraction of the samples (features)
    max_region_items: int = 20000     # no feature regions above this many features


@dataclass
class Frozen:
    """A chain that has finished: its final state still serves as a donor."""

    Z: np.ndarray
    U: np.ndarray
    free: np.ndarray


# --------------------------------------------------------------------------- data regions
def _clusters(B, min_size, max_size):
    """Index arrays of the clusters of an average-linkage tree of the rows of B (Jaccard
    distance), with sizes in [min_size, max_size]."""
    N = B.shape[0]
    if N < 2:
        return []
    Bf = B.astype(np.float32)
    inter = Bf @ Bf.T
    s = np.diag(inter).copy()
    union = s[:, None] + s[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(union > 0, 1.0 - inter / union, 1.0).astype(np.float64)
    iu = np.triu_indices(N, 1)
    tree = linkage(D[iu], "average")
    del D, inter, union
    members = {}
    out = []
    for t, (a, b, _h, size) in enumerate(tree):
        a, b = int(a), int(b)
        left = members.pop(a, None) if a >= N else np.array([a], np.int32)
        right = members.pop(b, None) if b >= N else np.array([b], np.int32)
        if size > max_size:            # its parents are larger still: never needed again
            continue
        both = np.concatenate([left, right])
        members[N + t] = both
        if size >= min_size:
            out.append(np.sort(both))
    return out


def data_regions(V, pcfg):
    """{"samples": [index arrays], "features": [index arrays]} from the observed ones of V."""
    B = V == 1
    n, F = B.shape
    out = {"samples": _clusters(B, pcfg.region_min_size,
                                max(pcfg.region_min_size, pcfg.region_max_fraction * n))}
    out["features"] = _clusters(B.T, pcfg.region_min_size,
                                max(pcfg.region_min_size, pcfg.region_max_fraction * F)) \
        if F <= pcfg.max_region_items else []
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- target pieces
def _log_bb(m, a, b, N):
    """log P(a column with m ones out of N) under Bernoulli(p), p ~ Beta(a, b)."""
    m = np.asarray(m, float)
    return (gammaln(a + m) + gammaln(b + N - m) - gammaln(a + b + N)
            - (gammaln(a) + gammaln(b) - gammaln(a + b)))


def _col_prior(member, mz, mu, n, F):
    """Prior (Beta-Bernoulli) of columns with mz carriers and mu members, summed."""
    return float(np.sum(_log_bb(mz, *member.zprior, n)) + np.sum(_log_bb(mu, *member.uprior, F)))


def _delta_ll(V, member, rows, cols, dC):
    """Change of the (tempered) log-likelihood of ``member`` when the counts of the block
    rows x cols change by dC (table rows as in ``kernels``: (i * rs) * G + fg[j])."""
    if rows.size == 0 or cols.size == 0:
        return 0.0
    eng = member.engine
    G = getattr(eng, "G", 1)
    rs = 1 if eng.T1.shape[0] > G else 0
    tr = (rows * rs * G)[:, None]
    if G > 1:
        tr = tr + eng.fg[cols][None, :]
    tr = np.broadcast_to(tr, (rows.size, cols.size))
    Vb = V[np.ix_(rows, cols)]
    C0 = eng.C[np.ix_(rows, cols)].astype(np.int64)
    C1 = C0 + dC
    new = np.where(Vb == 1, eng.T1[tr, C1], eng.T0[tr, C1])
    old = np.where(Vb == 1, eng.T1[tr, C0], eng.T0[tr, C0])
    return float(np.sum(np.where(Vb >= 0, new - old, 0.0)))


def _apply(member, rows, cols, dC):
    if rows.size and cols.size:
        idx = np.ix_(rows, cols)
        member.engine.C[idx] = (member.engine.C[idx] + dC).astype(member.engine.C.dtype)


def _used(m):
    return np.flatnonzero(m.free & m.Z.any(0) & m.U.any(0))


def _empty(m):
    return np.flatnonzero(m.free & ~m.Z.any(0) & ~m.U.any(0))


def _key(m, k):
    return np.packbits(m.Z[:, k]).tobytes() + b"|" + np.packbits(m.U[:, k]).tobytes()


def _log_perm(a, k):
    """log of a! / (a - k)!: the number of ordered choices of k slots out of a."""
    return float(gammaln(a + 1) - gammaln(a - k + 1))


# --------------------------------------------------------------------------- moves
def transplant_or_delete(V, rec, donor, rng, stats):
    """One transplant or delete proposal (probability 1/2 each) on ``rec`` from ``donor``."""
    n, F = V.shape
    D = _used(donor)
    if D.size == 0:
        return
    if rng.random() < 0.5:
        move = TRANSPLANT
        E = _empty(rec)
        if E.size == 0:
            return
        d = D[rng.integers(D.size)]
        e = E[rng.integers(E.size)]
        key = _key(donor, d)
        dkeys = [_key(donor, x) for x in D]
        c_d = sum(k == key for k in dkeys)
        dset = set(dkeys)
        n_match = sum(_key(rec, k) in dset for k in _used(rec)) + 1     # after the transplant
        z, u = donor.Z[:, d], donor.U[:, d]
        rows, cols = np.flatnonzero(z), np.flatnonzero(u)
        delta = (_delta_ll(V, rec, rows, cols, 1)
                 + _col_prior(rec, rows.size, cols.size, n, F) - _col_prior(rec, 0, 0, n, F))
        log_r = delta - np.log(n_match) - np.log(c_d / D.size) + np.log(E.size)
    else:
        move = DELETE
        dkeys = {}
        for x in D:
            k = _key(donor, x)
            dkeys[k] = dkeys.get(k, 0) + 1
        M = [k for k in _used(rec) if _key(rec, k) in dkeys]
        if not M:
            return
        e = M[rng.integers(len(M))]
        c_d = dkeys[_key(rec, e)]
        n_empty_after = _empty(rec).size + 1
        rows, cols = np.flatnonzero(rec.Z[:, e]), np.flatnonzero(rec.U[:, e])
        delta = (_delta_ll(V, rec, rows, cols, -1)
                 + _col_prior(rec, 0, 0, n, F) - _col_prior(rec, rows.size, cols.size, n, F))
        log_r = delta + np.log(c_d / D.size) - np.log(n_empty_after) + np.log(len(M))
    stats[0, move] += 1
    if log_r >= 0 or rng.random() < np.exp(log_r):
        if move == TRANSPLANT:
            rec.Z[:, e] = z
            rec.U[:, e] = u
            _apply(rec, rows, cols, 1)
        else:
            _apply(rec, rows, cols, -1)
            rec.Z[:, e] = 0
            rec.U[:, e] = 0
        stats[1, move] += 1


def _inside(m, region, kind):
    """Used slots of m whose carriers (kind "samples") or members ("features") all lie in the
    region."""
    used = _used(m)
    X = m.Z if kind == "samples" else m.U
    if used.size == 0:
        return used
    total = X[:, used].sum(0)
    inside = X[region][:, used].sum(0)
    return used[inside == total]


def crossover(V, a, b, kind, region, rng, stats):
    """Swap every component of chains a and b that lies inside ``region``."""
    n, F = V.shape
    move = CROSS_SAMPLES if kind == "samples" else CROSS_FEATURES
    Sa, Sb = _inside(a, region, kind), _inside(b, region, kind)
    if Sa.size == 0 and Sb.size == 0:
        return
    if sorted(_key(a, k) for k in Sa) == sorted(_key(b, k) for k in Sb):
        return                                   # the same components: nothing would change
    Ea, Eb = _empty(a), _empty(b)
    avail_a, avail_b = Sa.size + Ea.size, Sb.size + Eb.size
    stats[0, move] += 1
    if avail_a < Sb.size or avail_b < Sa.size:
        return                                   # no room: rejected
    Za, Ua, Zb, Ub = a.Z[:, Sa], a.U[:, Sa], b.Z[:, Sb], b.U[:, Sb]
    rows = np.flatnonzero(Za.any(1) | Zb.any(1))
    cols = np.flatnonzero(Ua.any(1) | Ub.any(1))
    dC = (Zb[rows].astype(np.int32) @ Ub[cols].T.astype(np.int32)
          - Za[rows].astype(np.int32) @ Ua[cols].T.astype(np.int32))
    delta = _delta_ll(V, a, rows, cols, dC) + _delta_ll(V, b, rows, cols, -dC)
    delta += (_col_prior(a, Zb.sum(0), Ub.sum(0), n, F) - _col_prior(a, Za.sum(0), Ua.sum(0), n, F)
              + (Sa.size - Sb.size) * _col_prior(a, 0, 0, n, F))
    delta += (_col_prior(b, Za.sum(0), Ua.sum(0), n, F) - _col_prior(b, Zb.sum(0), Ub.sum(0), n, F)
              + (Sb.size - Sa.size) * _col_prior(b, 0, 0, n, F))
    # slots: a uniformly random ordered choice among the freed and empty slots, both ways
    log_q = (_log_perm(avail_a, Sb.size) + _log_perm(avail_b, Sa.size)
             - _log_perm(avail_a, Sa.size) - _log_perm(avail_b, Sb.size))
    slots_a = rng.permutation(np.concatenate([Sa, Ea]))[:Sb.size]
    slots_b = rng.permutation(np.concatenate([Sb, Eb]))[:Sa.size]
    log_r = delta + log_q
    if log_r >= 0 or rng.random() < np.exp(log_r):
        a.Z[:, Sa] = 0
        a.U[:, Sa] = 0
        b.Z[:, Sb] = 0
        b.U[:, Sb] = 0
        a.Z[:, slots_a] = Zb
        a.U[:, slots_a] = Ub
        b.Z[:, slots_b] = Za
        b.U[:, slots_b] = Ua
        _apply(a, rows, cols, dC)
        _apply(b, rows, cols, -dC)
        stats[1, move] += 1


def population_round(V, members, active, regions, pcfg, rng, stats):
    """Transplant/delete proposals for every running chain (donor: a uniformly chosen other
    chain), then crossover proposals between random pairs of running chains."""
    chains = sorted(members)
    if len(chains) < 2:
        return
    for i in sorted(active):
        others = [c for c in chains if c != i]
        for _ in range(pcfg.n_transplant):
            j = others[rng.integers(len(others))]
            transplant_or_delete(V, members[i], members[j], rng, stats)
    run = sorted(active)
    kinds = sorted(regions)
    if len(run) < 2 or not kinds:
        return
    for _ in range(pcfg.n_crossover):
        i, j = rng.choice(len(run), 2, replace=False)
        kind = kinds[rng.integers(len(kinds))]
        region = regions[kind][rng.integers(len(regions[kind]))]
        crossover(V, members[run[i]], members[run[j]], kind, region, rng, stats)


# --------------------------------------------------------------------------- driver
def _population_key(cfg, seeds, pcfg):
    return repr((_config_key(cfg), [int(s) for s in seeds], vars(pcfg)))


def run_population(V, cfg, seeds, inits, pcfg, seed):
    """Run the chains in lockstep with population moves; returns (results, move stats).

    With ``cfg.checkpoint_dir`` the whole population (every chain, the moves' random state and
    finished chains' results) is saved to one file every ``cfg.checkpoint_every`` sweeps and a
    rerun resumes from it.
    """
    rng = np.random.default_rng(seed)
    stats = np.zeros((2, N_MOVES), np.int64)
    regions = data_regions(V, pcfg)
    n_chains = len(seeds)
    results, frozen, saved, sweep = {}, {}, {}, 0
    path = os.path.join(cfg.checkpoint_dir, "population.pkl") if cfg.checkpoint_dir else None
    key = _population_key(cfg, seeds, pcfg)
    if path is not None and os.path.exists(path):
        with open(path, "rb") as fh:
            ck = pickle.load(fh)
        if ck.get("format") != _FORMAT or ck.get("key") != key:
            raise ValueError(f"checkpoint {path} was written for different data or settings; "
                             "remove it or use another checkpoint_dir.")
        rng, stats, sweep = ck["rng"], ck["stats"], ck["sweep"]
        results, frozen, saved = ck["results"], ck["frozen"], ck["chains"]
    members = {c: Frozen(*f) for c, f in frozen.items()}
    steps = {c: chain_steps(V, cfg, seeds[c], inits[c], population=True, resume=saved.get(c))
             for c in range(n_chains) if c not in results}
    while steps:
        states = {}
        for c in list(steps):
            while True:
                try:
                    kind, obj = next(steps[c])
                except StopIteration as done:
                    results[c] = done.value
                    m = members.get(c)
                    if m is not None:
                        frozen[c] = (m.Z, m.U, m.free)
                        members[c] = Frozen(m.Z, m.U, m.free)
                    del steps[c]
                    break
                if kind == "checkpoint":
                    states[c] = obj
                    continue
                members[c] = obj
                break
        if states and path is not None:
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({"format": _FORMAT, "key": key, "rng": rng, "stats": stats,
                             "sweep": sweep, "results": results, "frozen": frozen,
                             "chains": states}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        sweep += 1
        if steps and sweep % pcfg.every == 0:
            population_round(V, members, set(steps), regions, pcfg, rng, stats)
    if path is not None and os.path.exists(path):
        os.remove(path)
    return [results[c] for c in range(n_chains)], stats
