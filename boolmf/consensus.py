"""Robust components from all posterior draws, and a convergence check on them.

On real data the posterior of a Boolean factorization has two layers. A robust layer of
components appears in most draws of every chain; a transient layer of small components differs
from draw to draw and chain to chain, and on large data keeps growing slowly long after the
robust layer has settled. The log-likelihood and the component count follow the transient layer,
so their R-hat can stay far above 1 while the robust components no longer change. These tools
work on the robust layer.

``robust_components`` finds the components present in most draws, by cell Jaccard similarity
(over the genome x gene rectangle a component covers) between components of different draws:

1. Candidates: the components of a few evenly spaced draws per chain. A component present in at
   least half of all draws is in a given draw with probability at least 1/2, so all candidate
   draws miss it with probability at most 2 ** -(n_chains * n_candidate_draws).
2. Support of a candidate: the share of all draws holding a component within ``match`` of it;
   it must be at least ``min_draw_share``, in at least ``min_chains`` chains.
3. Candidates within ``match`` of each other are one component; the best-supported represents
   it.
4. In every draw the component closest to the representative (if within ``match``) is its
   occurrence. Membership and activation probabilities are averaged over the occurrences.

``robust_stability`` splits every chain's draws into consecutive windows, finds the robust
components of each window, and reports how many of the last window's robust components were
already robust in each earlier window. Values near 1 mean the robust layer has settled.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["RobustComponents", "robust_components", "robust_stability"]


@dataclass
class RobustComponents:
    """Robust components and their posterior probabilities.

    Attributes
    ----------
    members : ndarray of shape (n_components, n_features)
        P(feature j belongs to component k | component k is present in the draw).
    activations : ndarray of shape (n_samples, n_components)
        P(sample i carries component k | component k is present in the draw).
    support : ndarray of shape (n_components,)
        Share of all draws in which component k is present.
    chain_support : ndarray of shape (n_components, n_chains)
        Share of each chain's draws in which component k is present.
    """

    members: np.ndarray
    activations: np.ndarray
    support: np.ndarray
    chain_support: np.ndarray

    @property
    def n_components(self):
        return self.members.shape[0]

    def coverage(self):
        """P(entry (i, j) is covered by at least one robust component), shape (n_samples,
        n_features): 1 - prod_k (1 - support_k * activations[i, k] * members[k, j]).

        This treats components, and memberships within a component, as independent; the exact
        posterior predictive averages over the draws instead.
        """
        log_none = np.zeros((self.activations.shape[0], self.members.shape[1]))
        for k in range(self.n_components):
            p = self.support[k] * np.outer(self.activations[:, k], self.members[k])
            log_none += np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12))
        return 1.0 - np.exp(log_none)

    def reconstruct(self, detection, background):
        """P(x[i, j] = 1) under ``or_flip``: background + (detection - background) * coverage.
        ``detection`` and ``background`` are floats or arrays of shape (n_samples,)."""
        a = np.asarray(detection, float).reshape(-1, 1)
        b = np.asarray(background, float).reshape(-1, 1)
        return b + (a - b) * self.coverage()


def _filter(Z, U, min_size):
    Z = np.asarray(Z, bool)
    U = np.asarray(U, bool)
    ok = (Z.sum(0) >= min_size) & (U.sum(0) >= min_size)
    return Z[:, ok].T.astype(np.float32), U[:, ok].T.astype(np.float32)


def _cell_jaccard(Ra, Ga, Rb, Gb):
    """Cell Jaccard between components a (rows Ra, Ga) and b: |A x B rectangles' overlap| /
    |union|, from the genome and gene sets."""
    inter = (Ra @ Rb.T) * (Ga @ Gb.T)
    size_a = Ra.sum(1) * Ga.sum(1)
    size_b = Rb.sum(1) * Gb.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(inter > 0, inter / (size_a[:, None] + size_b[None, :] - inter), 0.0)


def _check(draws):
    if len(draws) == 0 or any(len(c) == 0 for c in draws):
        raise ValueError("draws needs at least one draw in every chain.")
    return [len(c) for c in draws]


def robust_components(draws, *, min_size=3, match=0.5, min_draw_share=0.5, min_chains=2,
                      n_candidate_draws=8):
    """Components present in most posterior draws (see the module docstring).

    Parameters
    ----------
    draws : sequence over chains of sequences of (Z, U)
        Z (n_samples, k) and U (n_features, k) boolean: carriers and members of the k
        components of one draw. The inner sequences may be lazy (any object with ``len`` and
        indexing), since draws are visited one at a time.
    min_size : int, default=3
        Components with fewer carriers or members are ignored.
    match : float, default=0.5
        Cell Jaccard similarity at which two components count as the same.
    min_draw_share : float, default=0.5
        Smallest share of all draws a robust component appears in.
    min_chains : int, default=2
        Smallest number of chains a robust component appears in (capped at the number of
        chains).

    Returns
    -------
    RobustComponents
        Ordered by support, highest first.
    """
    lengths = _check(draws)
    n_chains = len(draws)
    min_chains = min(int(min_chains), n_chains)
    # 1. candidates
    cz, cu = [], []
    for c, chain in enumerate(draws):
        pick = np.unique(np.linspace(0, lengths[c] - 1, min(n_candidate_draws, lengths[c]))
                         .round().astype(int))
        for t in pick:
            R, G = _filter(*chain[t], min_size)
            cz.append(R)
            cu.append(G)
    n = draws[0][0][0].shape[0]
    F = draws[0][0][1].shape[0]
    Rc = np.concatenate(cz) if cz else np.zeros((0, n), np.float32)
    Gc = np.concatenate(cu) if cu else np.zeros((0, F), np.float32)
    empty = RobustComponents(np.zeros((0, F)), np.zeros((n, 0)), np.zeros(0),
                             np.zeros((0, n_chains)))
    if len(Rc) == 0:
        return empty
    # 2. support of every candidate
    hits = [np.zeros((len(Rc), m), bool) for m in lengths]
    for c, chain in enumerate(draws):
        for t in range(lengths[c]):
            R, G = _filter(*chain[t], min_size)
            if len(R):
                hits[c][:, t] = (_cell_jaccard(Rc, Gc, R, G) >= match).any(1)
    chain_share = np.stack([h.mean(1) for h in hits], 1)
    support = np.concatenate(hits, 1).mean(1)
    ok = np.flatnonzero((support >= min_draw_share) & ((chain_share > 0).sum(1) >= min_chains))
    if ok.size == 0:
        return empty
    # 3. group candidates that are the same component
    order = ok[np.argsort(-support[ok], kind="stable")]
    J = _cell_jaccard(Rc[order], Gc[order], Rc[order], Gc[order])
    reps = []
    for i in range(len(order)):
        if not any(J[i, j] >= match for j in reps):
            reps.append(i)
    Rr, Gr = Rc[order[reps]], Gc[order[reps]]
    K = len(reps)
    # 4. occurrences in every draw
    msum = np.zeros((K, F))
    asum = np.zeros((K, n))
    count = [np.zeros((K, m), bool) for m in lengths]
    for c, chain in enumerate(draws):
        for t in range(lengths[c]):
            R, G = _filter(*chain[t], min_size)
            if not len(R):
                continue
            Jd = _cell_jaccard(Rr, Gr, R, G)
            best = Jd.argmax(1)
            found = Jd[np.arange(K), best] >= match
            msum[found] += G[best[found]]
            asum[found] += R[best[found]]
            count[c][found, t] = True
    occ = np.concatenate(count, 1).sum(1)
    keep = occ > 0
    support_r = np.concatenate(count, 1).mean(1)
    chain_r = np.stack([x.mean(1) for x in count], 1)
    out = RobustComponents(
        members=(msum[keep] / occ[keep, None]),
        activations=(asum[keep] / occ[keep, None]).T,
        support=support_r[keep],
        chain_support=chain_r[keep])
    srt = np.argsort(-out.support, kind="stable")
    return RobustComponents(out.members[srt], out.activations[:, srt], out.support[srt],
                            out.chain_support[srt])


class _Window:
    def __init__(self, chain, lo, hi):
        self.chain, self.lo, self.hi = chain, lo, hi

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, t):
        return self.chain[self.lo + t]


def _binary(rc):
    return (rc.activations >= 0.5).T.astype(np.float32), (rc.members >= 0.5).astype(np.float32)


def robust_stability(draws, *, n_windows=4, **kwargs):
    """How much the robust components change over the run.

    Every chain's draws are split into ``n_windows`` consecutive windows of equal length, and
    ``robust_components`` (with ``kwargs``) is run on each window (all chains). Components are
    compared through their majority sets (probability at least 0.5) at the same ``match``.

    Returns
    -------
    list of dict, one per window, in order
        ``window``, ``n_robust``, ``found_in_last`` (how many of the last window's robust
        components are robust here), ``share_of_last`` (that count over the last window's),
        ``kept_in_last`` (how many of this window's robust components are robust in the last
        window) and ``median_jaccard_to_last`` (median over the last window's components of
        the best cell Jaccard to this window's).
    """
    lengths = _check(draws)
    if n_windows < 1 or min(lengths) < n_windows:
        raise ValueError(f"n_windows must be between 1 and the shortest chain ({min(lengths)}).")
    match = kwargs.get("match", 0.5)
    res = []
    for w in range(n_windows):
        sub = [_Window(chain, w * m // n_windows, (w + 1) * m // n_windows)
               for chain, m in zip(draws, lengths)]
        res.append(robust_components(sub, **kwargs))
    Rl, Gl = _binary(res[-1])
    out = []
    for w, rc in enumerate(res):
        R, G = _binary(rc)
        if len(R) and len(Rl):
            J = _cell_jaccard(Rl, Gl, R, G)
            to_last, from_last = J.max(1), J.max(0)
        else:
            to_last, from_last = np.zeros(len(Rl)), np.zeros(len(R))
        found = int((to_last >= match).sum())
        out.append(dict(window=w, n_robust=rc.n_components, found_in_last=found,
                        share_of_last=found / len(Rl) if len(Rl) else float("nan"),
                        kept_in_last=int((from_last >= match).sum()),
                        median_jaccard_to_last=float(np.median(to_last)) if len(Rl)
                        else float("nan")))
    return out
