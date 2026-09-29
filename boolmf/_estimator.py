"""The BoolMF estimator."""

import numbers
import os
import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._model import (
    LIKELIHOODS,
    PositivePrior,
    RatePrior,
    beta_params_or_raise,
    loglik_tables,
)
from ._sampler.chain import ChainConfig, run_chain
from ._sampler.kernels import project_activations
from ._sampler.splitmerge import MOVE_NAMES
from .diagnostics import ess, rhat
from .matching import match_components
from .utils.validation import row_seeds, to_binary_int8

DEFAULT_SPLIT_MERGE = 10          # proposals per sweep when split_merge=True

__all__ = ["BoolMF", "AnchorComponent"]


@dataclass
class AnchorComponent:
    """A component that is active in every sample.

    Parameters
    ----------
    members : array-like
        Feature indices, or a boolean mask of length n_features.
    membership : {"fixed", "learned"}, default="fixed"
        ``"fixed"`` keeps the members as given; ``"learned"`` starts from them and lets the
        sampler add or drop members.
    """

    members: object
    membership: str = "fixed"


def _two_means_threshold(x):
    """Split a 1-D array into two groups by 2-means and return the midpoint threshold."""
    x = np.asarray(x, float)
    if x.size == 0 or np.all(x == x[0]):
        return np.inf
    lo, hi = x.min(), x.max()
    t = (lo + hi) / 2
    for _ in range(50):
        a, b = x[x <= t], x[x > t]
        if a.size == 0 or b.size == 0:
            break
        t_new = (a.mean() + b.mean()) / 2
        if abs(t_new - t) < 1e-12:
            break
        t = t_new
    return t


def _bayes_fdr_threshold(p, fdr):
    """Smallest threshold whose calls have expected FDR = mean(1 - p) <= fdr."""
    s = np.sort(np.asarray(p, float).ravel())[::-1]
    if s.size == 0:
        return 1.0
    cum = np.cumsum(1.0 - s) / np.arange(1, s.size + 1)
    ok = np.flatnonzero(cum <= fdr)
    return float(s[ok[-1]]) if ok.size else 1.0


class BoolMF(TransformerMixin, BaseEstimator):
    """Bayesian nonparametric Boolean matrix factorization.

    Models a binary matrix X (samples x features) as the union of latent components: each
    component has binary members (features) and is active or inactive in each sample, and a
    feature is present when an active component containing it transmits it, or through
    background presence. With ``n_components=None`` the number of components is given an
    Indian buffet process prior and learned from the data. The posterior is sampled with
    several independent Gibbs chains whose components are matched with the Hungarian
    algorithm and pooled.

    Parameters
    ----------
    n_components : int or None, default=None
        None learns the number of components (Indian buffet process); an int fixes it.
    max_components : int or "auto", default="auto"
        Number of component slots (truncation of the Indian buffet process).
        ``"auto"`` uses ``max(1, min(n_samples, n_features) // 2)``. A warning is raised when
        more than 80% of the slots are in use, since the cap then limits the rank.
    likelihood : {"or_flip", "noisy_or"}, default="or_flip"
        ``"or_flip"``: P(x = 1) = detection if c >= 1, else background, where c is the number
        of active components containing the feature. ``"noisy_or"``: P(x = 1) =
        1 - (1 - background) (1 - detection)^c, so a feature in two active components is more
        likely observed. The two agree whenever c <= 1. With ``"noisy_or"`` a component whose
        members are split across two components with the same carriers is a local mode that
        one-at-a-time updates rarely leave; split-merge moves (planned for v0.2) address this,
        so ``"or_flip"`` is the default for now.
    anchor_components : list of AnchorComponent or None, default=None
        Components active in every sample, for example a core-genome component.
    binarize : float or None, default=None
        None requires 0/1 input (NaN = missing). A float maps values above it to 1.
    detection_prior, background_prior : scipy.stats frozen distribution, float or None
        Priors on the two rates. None is Beta(1, 1). A float in (0, 1) fixes the rate. Any
        distribution with support in [0, 1] is accepted.
    membership_prior : scipy.stats.beta frozen distribution or None, default=None
        Prior on each component's membership rate. None is Beta(1, 1).
    alpha_prior : scipy.stats.gamma frozen distribution or None, default=None
        Prior on the Indian buffet process concentration. None is Gamma(1, 1).
    n_chains : int, default=20
        Number of independent chains.
    max_sweeps : int, default=10000
        Upper bound on sweeps per chain (burn-in plus sampling).
    burn_in : int or "auto", default="auto"
        ``"auto"`` ends burn-in once the chain's recent trace passes a segment R-hat check on
        the component count and rates (cut-off ``max(1.05, rhat_threshold)``) and a Geweke
        check on the log-likelihood twice in a row; an int fixes the burn-in length.
    n_draws : int, default=100
        Posterior draws kept per chain.
    thin : int or "auto", default="auto"
        Sweeps between kept draws. ``"auto"`` uses the autocorrelation of the log-likelihood.
    split_merge : bool or int, default=True
        Metropolis–Hastings moves on pairs of components after each Gibbs sweep. Each proposes
        splitting a component in two, merging two, reallocating samples and features between
        two (restricted Gibbs proposals, Jain & Neal 2004), or rewriting two components without
        changing which entries they cover (moving shared features into the component whose
        carriers contain the other's). They let chains leave states that single-variable Gibbs
        updates cannot, such as two components merged into one. True makes 10 proposals per
        sweep, an int sets the number, False (or 0) turns them off.
    rhat_threshold : float, default=1.01
        Convergence cut-off. A warning is raised when the cross-chain R-hat of the detection or
        background rate exceeds ``max(1.05, rhat_threshold)``.
    min_ess : float, default=400
        A warning is raised when the pooled effective sample size of the detection or
        background rate is below this value (checked when ``n_chains * n_draws`` is at least
        twice ``min_ess``). The log-likelihood and component count are reported in ``ess_``
        but do not trigger warnings: they mix slowly because short-lived components that
        absorb noise come and go, which leaves robust components unaffected.
    init : {"random", "nmf"} or tuple of (members, activations), default="random"
        Starting state for the chains. ``"nmf"`` fits ``sklearn.decomposition.NMF(**init_params)``
        and binarizes each component. A tuple gives members (n_init, n_features) and
        activations (n_samples, n_init) directly.
    init_params : dict or None, default=None
        Keyword arguments for ``sklearn.decomposition.NMF`` when ``init="nmf"``.
    robustness_threshold : float, default=0.5
        Fraction of chains a component must appear in to be flagged robust.
    min_support : float, default=3
        Components with fewer expected members or active samples are flagged low-support.
    store_draws : bool, default=True
        Keep bit-packed binary draws (needed by ``transform`` and ``get_draws``).
    random_state : int, RandomState instance or None, default=None
    n_jobs : int or None, default=None
        Number of chains run in parallel (joblib).
    verbose : int, default=0

    Attributes
    ----------
    components_ : ndarray of shape (n_components_total, n_features)
        Posterior membership probabilities, P(feature j is a member of component k).
    activations_ : ndarray of shape (n_samples, n_components_total)
        Posterior activation probabilities, P(component k is active in sample i).
    n_components_ : int
        Number of robust, well-supported components.
    component_flags_ : ndarray of str
        ``"anchor"``, ``"robust"``, ``"low_support"`` or ``"not_robust"`` per component.
    robustness_ : ndarray
        Fraction of good chains in which each component appears.
    prevalence_ : ndarray
        Mean activation of each component across samples.
    detection_rate_, background_rate_ : float
        Posterior means of the two rates.
    alpha_ : float or None
        Posterior mean of the Indian buffet process concentration.
    n_components_draws_ : ndarray of shape (n_good_chains, n_draws)
        Posterior draws of the number of active (non-anchor) components.
    rhat_, ess_ : dict
        Cross-chain R-hat and effective sample size of the monitored scalars.
    split_merge_acceptance_ : dict
        Share of proposals accepted per move type (split, merge, reallocate, factor,
        unfactor), pooled over chains.
    log_likelihood_trace_ : ndarray of shape (n_chains, max_sweeps_run)
        Log-likelihood per sweep, NaN-padded.
    chain_status_ : ndarray of str
        ``"ok"``, ``"not_converged"`` or ``"stuck"`` per chain.
    n_iter_ : int
        Largest number of sweeps run by a chain.
    n_features_in_ : int
    feature_names_in_ : ndarray of str
        Defined when X has string column names.
    """

    def __init__(
        self,
        n_components=None,
        *,
        max_components="auto",
        likelihood="or_flip",
        anchor_components=None,
        binarize=None,
        detection_prior=None,
        background_prior=None,
        membership_prior=None,
        alpha_prior=None,
        n_chains=20,
        max_sweeps=10000,
        burn_in="auto",
        n_draws=100,
        thin="auto",
        split_merge=True,
        rhat_threshold=1.01,
        min_ess=400,
        init="random",
        init_params=None,
        robustness_threshold=0.5,
        min_support=3,
        store_draws=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.max_components = max_components
        self.likelihood = likelihood
        self.anchor_components = anchor_components
        self.binarize = binarize
        self.detection_prior = detection_prior
        self.background_prior = background_prior
        self.membership_prior = membership_prior
        self.alpha_prior = alpha_prior
        self.n_chains = n_chains
        self.max_sweeps = max_sweeps
        self.burn_in = burn_in
        self.n_draws = n_draws
        self.thin = thin
        self.split_merge = split_merge
        self.rhat_threshold = rhat_threshold
        self.min_ess = min_ess
        self.init = init
        self.init_params = init_params
        self.robustness_threshold = robustness_threshold
        self.min_support = min_support
        self.store_draws = store_draws
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ sklearn plumbing
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        return tags

    def _n_split_merge(self):
        if isinstance(self.split_merge, (bool, np.bool_)):
            return DEFAULT_SPLIT_MERGE if self.split_merge else 0
        return int(self.split_merge)

    def _check_params(self):
        def _int(name, lo):
            v = getattr(self, name)
            if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < lo:
                raise ValueError(f"{name} must be an int >= {lo}; got {v!r}.")

        if self.n_components is not None:
            _int("n_components", 1)
        if self.max_components != "auto":
            _int("max_components", 1)
        if self.likelihood not in LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {LIKELIHOODS}; got {self.likelihood!r}.")
        _int("n_chains", 1)
        _int("max_sweeps", 2)
        _int("n_draws", 1)
        if not isinstance(self.split_merge, (bool, np.bool_)):
            _int("split_merge", 0)
        if self.burn_in != "auto":
            _int("burn_in", 0)
        if self.thin != "auto":
            _int("thin", 1)
        for name in ("rhat_threshold", "min_ess", "robustness_threshold", "min_support"):
            v = getattr(self, name)
            if not isinstance(v, numbers.Real) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} must be a non-negative number; got {v!r}.")
        if not (isinstance(self.init, tuple) or self.init in ("random", "nmf")):
            raise ValueError("init must be 'random', 'nmf' or a (members, activations) tuple.")
        if self.init_params is not None and not isinstance(self.init_params, dict):
            raise ValueError("init_params must be a dict or None.")
        if self.binarize is not None and not isinstance(self.binarize, numbers.Real):
            raise ValueError("binarize must be a number or None.")

    def _validate_X(self, X, mask, reset):
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), ensure_all_finite="allow-nan",
            dtype="numeric", reset=reset,
        )
        return to_binary_int8(X, mask=mask, binarize=self.binarize)

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y=None, mask=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Binary data; NaN marks missing entries.
        y : Ignored
        mask : array-like of bool of shape (n_samples, n_features), optional
            True marks entries to ignore (for example held-out entries).

        Returns
        -------
        self
        """
        self._check_params()
        V = self._validate_X(X, mask, reset=True)
        n, F = V.shape
        if not (V >= 0).any():
            raise ValueError("X has no observed entries.")

        anchors, anchor_learned = self._anchor_masks(F)
        na = len(anchors)
        if self.n_components is not None:
            n_free = int(self.n_components)
            nonparametric = False
        else:
            n_free = (max(1, min(n, F) // 2) if self.max_components == "auto"
                      else int(self.max_components))
            nonparametric = True
        n_slots = na + n_free
        n_init = n_free if not nonparametric else int(
            min(n_free, max(2, np.ceil(2 * np.sqrt(min(n, F))))))

        rs = check_random_state(self.random_state)
        base_seed = int(rs.randint(0, 2**31 - 1))
        ss = np.random.SeedSequence(base_seed)
        chain_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(self.n_chains)]
        self._transform_seed_ = int(np.random.SeedSequence(base_seed + 1).generate_state(1)[0])

        init = self._initial_state(V, n_free)
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, min(n_jobs, self.n_chains))
        n_threads = max(1, (os.cpu_count() or 1) // max(1, n_workers)) if n_workers > 1 else 0

        cfg = ChainConfig(
            likelihood=self.likelihood,
            n_slots=n_slots,
            anchor_members=anchors,
            anchor_learned=anchor_learned,
            nonparametric=nonparametric,
            prior_a=RatePrior.from_param(self.detection_prior, "detection_prior"),
            prior_b=RatePrior.from_param(self.background_prior, "background_prior"),
            membership_ab=beta_params_or_raise(self.membership_prior, "membership_prior"),
            alpha_prior=PositivePrior.from_param(self.alpha_prior, "alpha_prior"),
            burn_in=self.burn_in if self.burn_in == "auto" else int(self.burn_in),
            max_sweeps=int(max(self.max_sweeps, self.n_draws + 1)),
            n_draws=int(self.n_draws),
            thin=self.thin if self.thin == "auto" else int(self.thin),
            rhat_threshold=float(self.rhat_threshold),
            n_init=n_init,
            store_draws=bool(self.store_draws),
            store_entries=True,
            n_threads=n_threads,
            min_support=int(np.ceil(self.min_support)),
            burn_rhat=max(1.05, float(self.rhat_threshold)),
            n_split_merge=self._n_split_merge(),
            verbose=self.verbose,
        )
        results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(run_chain)(V, cfg, s, init) for s in chain_seeds
        )
        self._postprocess(V, results, na, nonparametric, n_free)
        return self

    def _anchor_masks(self, F):
        anchors, learned = [], []
        for a in self.anchor_components or []:
            if not isinstance(a, AnchorComponent):
                raise TypeError("anchor_components must be a list of AnchorComponent.")
            m = np.asarray(a.members)
            if m.dtype == bool:
                if m.shape != (F,):
                    raise ValueError("a boolean anchor mask must have length n_features.")
                mask = m.astype(np.int8)
            else:
                mask = np.zeros(F, np.int8)
                mask[m.astype(int)] = 1
            if a.membership not in ("fixed", "learned"):
                raise ValueError("AnchorComponent.membership must be 'fixed' or 'learned'.")
            anchors.append(mask)
            learned.append(a.membership == "learned")
        return anchors, learned

    def _initial_state(self, V, n_free):
        if isinstance(self.init, tuple):
            members, activations = (np.asarray(x) for x in self.init)
            if members.ndim != 2 or members.shape[1] != V.shape[1]:
                raise ValueError("init members must have shape (n_init, n_features).")
            if activations.shape != (V.shape[0], members.shape[0]):
                raise ValueError("init activations must have shape (n_samples, n_init).")
            if members.shape[0] > n_free:
                warnings.warn("init has more components than slots; extra ones are dropped.",
                              stacklevel=3)
            return (members.astype(bool), activations.astype(bool))
        if self.init == "nmf":
            from sklearn.decomposition import NMF

            params = dict(self.init_params or {})
            params.setdefault("n_components", min(n_free, max(2, int(np.sqrt(min(V.shape))))))
            params.setdefault("init", "nndsvda")
            params.setdefault("max_iter", 500)
            if "random_state" not in params and self.random_state is not None:
                params["random_state"] = self.random_state
            Xf = np.where(V > 0, 1.0, 0.0).astype(np.float32)
            nmf = NMF(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W = nmf.fit_transform(Xf)
            H = nmf.components_
            members = np.stack([H[k] > _two_means_threshold(H[k]) for k in range(H.shape[0])])
            acts = np.stack([W[:, k] > _two_means_threshold(W[:, k]) for k in range(W.shape[1])],
                            axis=1)
            return (members, acts)
        return "random"

    def _postprocess(self, V, results, na, nonparametric, n_free):
        n, F = V.shape
        n_chains = len(results)
        # ---- chain status -----------------------------------------------------------------
        ll_mean = np.array([r.draw_loglik.mean() for r in results])
        ll_sd = np.array([r.draw_loglik.std() for r in results])
        scale = max(np.median(ll_sd), 1e-6 * max(1.0, abs(np.median(ll_mean))))
        best_half = np.sort(ll_mean)[n_chains // 2:]
        ref_level = np.median(best_half)
        status = np.array(["ok" if r.converged else "not_converged" for r in results], dtype=object)
        stuck = (ref_level - ll_mean) > 10 * scale
        status[stuck] = "stuck"
        good = np.flatnonzero(status != "stuck")
        if good.size == 0:
            good = np.arange(n_chains)
        self.chain_status_ = status.astype(str)

        # ---- diagnostics across good chains -----------------------------------------------
        L = min(len(results[c].draw_loglik) for c in good)
        traces = {
            "log_likelihood": np.stack([results[c].draw_loglik[-L:] for c in good]),
            "n_active": np.stack([results[c].draw_n_active[-L:] for c in good]).astype(float),
            "detection_rate": np.stack([results[c].draw_rates[-L:, 0] for c in good]),
            "background_rate": np.stack([results[c].draw_rates[-L:, 1] for c in good]),
        }
        if nonparametric:
            traces["alpha"] = np.stack([results[c].draw_alpha[-L:] for c in good])
        self.rhat_ = {k: rhat(v) for k, v in traces.items()}
        sm = np.sum([r.split_merge for r in results], axis=0)
        self.split_merge_acceptance_ = {
            name: float(sm[1, t] / sm[0, t]) if sm[0, t] else float("nan")
            for t, name in enumerate(MOVE_NAMES)}
        self.ess_ = {k: ess(v) for k, v in traces.items()}
        # The log-likelihood and component count mix slowly because short-lived components
        # that absorb noise come and go; they are reported but do not trigger warnings.
        rate_keys = ("detection_rate", "background_rate")
        bad_rhat = [k for k in rate_keys if self.rhat_[k] > max(self.rhat_threshold, 1.05)]
        low_ess = [k for k in rate_keys if self.ess_[k] < self.min_ess]
        if bad_rhat and len(good) > 1:
            warnings.warn(
                "chains disagree on " + " and ".join(bad_rhat) + " (R-hat "
                + ", ".join(f"{self.rhat_[k]:.3f}" for k in bad_rhat) + "); they may sit in "
                "different modes. Check chain_status_ and consider more sweeps.", stacklevel=3)
        elif low_ess and len(good) * L >= 2 * self.min_ess:
            warnings.warn(
                "pooled ESS of " + " and ".join(low_ess) + " is "
                + ", ".join(f"{self.ess_[k]:.0f}" for k in low_ess)
                + f" (< min_ess={self.min_ess}); consider more draws or chains.", stacklevel=3)
        maxlen = max(len(r.trace["log_likelihood"]) for r in results)
        self.log_likelihood_trace_ = np.full((n_chains, maxlen), np.nan)
        for c, r in enumerate(results):
            ll_c = r.trace["log_likelihood"]
            self.log_likelihood_trace_[c, :len(ll_c)] = ll_c
        self.n_iter_ = int(maxlen)
        self.n_components_draws_ = traces["n_active"].astype(int)

        # ---- match components across good chains -----------------------------------------
        order = good[np.argsort(-ll_mean[good])]
        pooled = []          # list of dict(set=bool F, members=[(chain, slot)])

        def comp_slots(r):
            m = r.member_mean[:, na:] >= 0.5
            a = r.activation_mean[:, na:] >= 0.5
            keep = np.flatnonzero((m.sum(0) >= 1) & (a.sum(0) >= 1))
            return keep + na, (m[:, keep].T if keep.size else np.zeros((0, F), bool))

        for c in order:
            slots, sets = comp_slots(results[c])
            if slots.size == 0:
                continue
            if pooled:
                ref = np.stack([p["set"] for p in pooled])
                pairs = match_components(ref, sets, threshold=0.5)
            else:
                pairs = []
            matched = set()
            for ri, oi, _ in pairs:
                pooled[ri]["members"].append((c, int(slots[oi])))
                matched.add(oi)
            for oi in range(len(slots)):
                if oi not in matched:
                    pooled.append({"set": sets[oi], "members": [(c, int(slots[oi]))]})

        n_good = len(good)
        comps, acts, robust, flags = [], [], [], []
        mapping = {int(c): {} for c in good}
        for a in range(na):
            comps.append(np.mean([results[c].member_mean[:, a] for c in good], axis=0))
            acts.append(np.ones(n))
            robust.append(1.0)
            flags.append("anchor")
            for c in good:
                mapping[int(c)][a] = a
        for p in pooled:
            k = len(comps)
            comps.append(np.mean([results[c].member_mean[:, s] for c, s in p["members"]], axis=0))
            acts.append(np.mean([results[c].activation_mean[:, s] for c, s in p["members"]],
                                axis=0))
            robust.append(len({c for c, _ in p["members"]}) / n_good)
            for c, s in p["members"]:
                mapping[int(c)][s] = k
            flags.append(None)
        comps = np.asarray(comps, float).reshape(len(comps), F)
        acts = np.asarray(acts, float).reshape(len(acts), n).T
        robust = np.asarray(robust, float)
        exp_members = comps.sum(1)
        exp_active = acts.sum(0)
        for k in range(na, len(flags)):
            if robust[k] < self.robustness_threshold:
                flags[k] = "not_robust"
            elif exp_members[k] < self.min_support or exp_active[k] < self.min_support:
                flags[k] = "low_support"
            else:
                flags[k] = "robust"
        flags = np.asarray(flags, dtype=object)
        prevalence = acts.mean(0) if acts.size else np.zeros(0)
        rank = {"anchor": 0, "robust": 1, "low_support": 2, "not_robust": 3}
        perm = sorted(range(len(flags)), key=lambda k: (rank[flags[k]], -prevalence[k], k))
        inv = np.empty(len(perm), int)
        inv[perm] = np.arange(len(perm))
        self.components_ = comps[perm]
        self.activations_ = acts[:, perm]
        self.robustness_ = robust[perm]
        self.component_flags_ = flags[perm].astype(str)
        self.prevalence_ = prevalence[perm]
        self.n_components_ = int((self.component_flags_ == "robust").sum())
        self._slot_map_ = {c: {s: int(inv[k]) for s, k in m.items()} for c, m in mapping.items()}

        rates = np.concatenate([results[c].draw_rates for c in good])
        self.detection_rate_ = float(rates[:, 0].mean())
        self.background_rate_ = float(rates[:, 1].mean())
        self.alpha_ = float(np.nanmean(np.concatenate([results[c].draw_alpha for c in good]))) \
            if nonparametric else None
        if nonparametric and np.median(self.n_components_draws_) > 0.8 * n_free:
            warnings.warn(
                "more than 80% of component slots are in use; increase max_components.",
                stacklevel=3)

        # ---- entry-level summaries and draws ---------------------------------------------
        self._explained_ = np.mean([results[c].explained for c in good], axis=0)
        self._predictive_ = np.mean([results[c].predictive for c in good], axis=0)
        self._train_V_ = V
        self._n_anchor_ = na
        self._draws_ = []
        if self.store_draws:
            for c in good:
                r = results[c]
                for d, (rate_a, rate_b) in zip(r.draws, r.draw_rates):
                    self._draws_.append({
                        "chain": int(c), "slots": d["slots"], "U": d["U"], "Z": d["Z"],
                        "pi": d["pi"], "a": float(rate_a), "b": float(rate_b),
                    })
        self._n_train_samples_ = n

    # ------------------------------------------------------------------ inference outputs
    def fit_transform(self, X, y=None, mask=None):
        """Fit and return the posterior activation probabilities of the training samples."""
        return self.fit(X, y, mask=mask).activations_

    def transform(self, X, mask=None):
        """Posterior activation probabilities for (new) samples, components held fixed.

        Each stored posterior draw of the memberships and rates is used in turn; activations
        are Gibbs-sampled per sample and averaged. A sample's result depends only on its own
        row, not on the other rows passed with it.

        Returns
        -------
        ndarray of shape (n_samples, n_components_total)
        """
        check_is_fitted(self, "components_")
        V = self._validate_X(X, mask, reset=False)
        K = self.components_.shape[0]
        n = V.shape[0]
        if not self._draws_:
            raise RuntimeError("transform needs stored draws; refit with store_draws=True.")
        F = V.shape[1]
        seeds = row_seeds(V, self._transform_seed_)
        out = np.zeros((n, K))
        cnt = np.zeros(K)
        draws = self._select_draws(max_per_chain=10)
        for di, d in enumerate(draws):
            slots = d["slots"]
            U = np.unpackbits(d["U"], axis=0, count=F).astype(np.int8)
            T1, T0 = loglik_tables(self.likelihood, d["a"], d["b"], U.shape[1])
            pi = np.clip(d["pi"], 1e-12, 1 - 1e-12)
            logit_pi = np.log(pi) - np.log1p(-pi)
            logit_pi[slots < self._n_anchor_] = 50.0
            k, j = np.nonzero(U.T)
            ptr = np.zeros(U.shape[1] + 1, np.int64)
            np.cumsum(np.bincount(k, minlength=U.shape[1]), out=ptr[1:])
            with np.errstate(over="ignore"):
                s = seeds ^ np.uint64((di + 1) * 0x9E3779B97F4A7C15 % 2**64)
            z = project_activations(V, U, T1, T0, logit_pi, ptr, j.astype(np.int64), s, 40, 20)
            smap = self._slot_map_[d["chain"]]
            present = np.zeros(K, bool)
            for col, slot in enumerate(slots):
                kk = smap.get(int(slot))
                if kk is not None:
                    out[:, kk] += z[:, col]
                    present[kk] = True
            cnt[present] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cnt > 0, out / np.maximum(cnt, 1), 0.0)
        return out

    def _select_draws(self, max_per_chain):
        by_chain = {}
        for d in self._draws_:
            by_chain.setdefault(d["chain"], []).append(d)
        sel = []
        for ds in by_chain.values():
            idx = np.unique(np.linspace(0, len(ds) - 1, min(max_per_chain, len(ds))).astype(int))
            sel.extend(ds[i] for i in idx)
        return sel

    def inverse_transform(self, X):
        """Presence probabilities implied by activation probabilities.

        Uses the posterior mean memberships and rates (treating components as independent).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components_total)
            Activation probabilities or 0/1 activations, for example from ``transform``.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            P(x_ij = 1).
        """
        check_is_fitted(self, "components_")
        Zp = np.asarray(X, float)
        if Zp.ndim != 2 or Zp.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"expected shape (n_samples, {self.components_.shape[0]}); got {Zp.shape}.")
        M = self.components_
        n, F = Zp.shape[0], M.shape[1]
        a, b = self.detection_rate_, self.background_rate_
        log_none = np.zeros((n, F))
        weight = a if self.likelihood == "noisy_or" else 1.0
        for k in range(M.shape[0]):
            q = np.clip(np.outer(Zp[:, k], M[k]) * weight, 0.0, 1.0 - 1e-12)
            log_none += np.log1p(-q)
        none = np.exp(log_none)
        if self.likelihood == "noisy_or":
            return 1.0 - (1.0 - b) * none
        return b + (a - b) * (1.0 - none)

    def _is_training(self, V):
        """True when V has the training shape and agrees with every observed training entry
        (entries masked during fitting may hold any value)."""
        T = self._train_V_
        if V.shape != T.shape:
            return False
        obs = T >= 0
        return bool(np.array_equal(V[obs], T[obs]))

    def explained_probability(self, X=None):
        """P(entry is explained by an active component), per sample and feature.

        With ``X=None`` this is the exact posterior average over draws for the training
        samples; for other samples it is computed from ``transform``.
        """
        check_is_fitted(self, "components_")
        if X is None:
            return self._explained_.astype(float)
        Zp = self.transform(X)
        M = self.components_
        log_none = np.zeros((Zp.shape[0], M.shape[1]))
        for k in range(M.shape[0]):
            log_none += np.log1p(-np.clip(np.outer(Zp[:, k], M[k]), 0, 1 - 1e-12))
        return 1.0 - np.exp(log_none)

    def predictive_probability(self, X=None):
        """Posterior predictive P(x_ij = 1) for every entry.

        ``X=None`` returns the posterior average over draws for the training samples
        (including entries masked during fitting); other samples are projected with
        ``transform``.
        """
        check_is_fitted(self, "components_")
        if X is None:
            return self._predictive_.astype(float)
        return self.inverse_transform(self.transform(X))

    def score(self, X, y=None, entries=None):
        """Mean log predictive density (higher is better).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
        entries : array-like of bool of shape (n_samples, n_features), optional
            Score only these entries of the training samples, using the posterior predictive
            from fitting (for entry-wise cross-validation, pass the entries masked at fit
            time). Without ``entries``, the samples are projected with ``transform``.
        """
        check_is_fitted(self, "components_")
        if entries is not None:
            entries = np.asarray(entries, bool)
            V = self._validate_X(X, None, reset=False)
            if V.shape[0] != self._n_train_samples_ or entries.shape != V.shape:
                raise ValueError("entries requires X to be the training matrix.")
            sel = entries & (V >= 0)
            P = np.clip(self._predictive_[sel], 1e-12, 1 - 1e-12)
            x = V[sel]
        else:
            V = self._validate_X(X, None, reset=False)
            P = np.clip(self.predictive_probability(X), 1e-12, 1 - 1e-12)
            sel = V >= 0
            P, x = P[sel], V[sel]
        if x.size == 0:
            return np.nan
        return float(np.mean(np.where(x == 1, np.log(P), np.log1p(-P))))

    def score_samples(self, X):
        """Per-sample mean log predictive density; low values flag unusual samples."""
        check_is_fitted(self, "components_")
        V = self._validate_X(X, None, reset=False)
        P = np.clip(self.predictive_probability(X), 1e-12, 1 - 1e-12)
        ll = np.where(V == 1, np.log(P), np.log1p(-P))
        ll[V < 0] = np.nan
        return np.nanmean(ll, axis=1)

    def binarize_components(self, threshold=0.5, method="threshold", fdr=0.01):
        """Binary membership and activation calls.

        Parameters
        ----------
        threshold : float, default=0.5
            Used when ``method="threshold"`` (Bayes rule under Hamming loss).
        method : {"threshold", "bfdr"}
            ``"bfdr"`` chooses, separately for memberships and activations, the smallest
            threshold whose calls have expected false discovery rate at most ``fdr``.
        fdr : float, default=0.01

        Returns
        -------
        members : ndarray of bool, shape (n_components_total, n_features)
        activations : ndarray of bool, shape (n_samples, n_components_total)
        """
        check_is_fitted(self, "components_")
        if method == "threshold":
            tm = ta = threshold
        elif method == "bfdr":
            tm = _bayes_fdr_threshold(self.components_, fdr)
            ta = _bayes_fdr_threshold(self.activations_, fdr)
        else:
            raise ValueError("method must be 'threshold' or 'bfdr'.")
        return self.components_ >= tm, self.activations_ >= ta

    def get_draws(self, kind="components"):
        """Iterate over posterior draws mapped onto the pooled components.

        Parameters
        ----------
        kind : {"components", "activations"}

        Yields
        ------
        ndarray of bool
            (n_components_total, n_features) or (n_train_samples, n_components_total).
        """
        check_is_fitted(self, "components_")
        K, F = self.components_.shape
        n = self._n_train_samples_
        for d in self._draws_:
            smap = self._slot_map_[d["chain"]]
            if kind == "components":
                U = np.unpackbits(d["U"], axis=0, count=F).astype(bool)
                out = np.zeros((K, F), bool)
                for col, slot in enumerate(d["slots"]):
                    k = smap.get(int(slot))
                    if k is not None:
                        out[k] |= U[:, col]
            elif kind == "activations":
                Z = np.unpackbits(d["Z"], axis=0, count=n).astype(bool)
                out = np.zeros((n, K), bool)
                for col, slot in enumerate(d["slots"]):
                    k = smap.get(int(slot))
                    if k is not None:
                        out[:, k] |= Z[:, col]
            else:
                raise ValueError("kind must be 'components' or 'activations'.")
            yield out

    def summary(self):
        """One row per component: flag, robustness, prevalence, support, integrity, leakage.

        Returns a pandas DataFrame when pandas is installed, otherwise a dict of arrays.
        """
        from .metrics import component_integrity, component_leakage

        check_is_fitted(self, "components_")
        data = {
            "component": np.arange(self.components_.shape[0]),
            "flag": self.component_flags_,
            "robustness": self.robustness_,
            "prevalence": self.prevalence_,
            "expected_members": self.components_.sum(1),
            "expected_active": self.activations_.sum(0),
            "integrity": component_integrity(self),
            "leakage": component_leakage(self),
        }
        try:
            import pandas as pd
        except ImportError:
            return data
        return pd.DataFrame(data).set_index("component")

    def get_feature_names_out(self, input_features=None):
        """Output names for ``transform``: ``boolmf0``, ``boolmf1``, ..."""
        check_is_fitted(self, "components_")
        return np.asarray([f"boolmf{i}" for i in range(self.components_.shape[0])], dtype=object)


BooleanMF = BoolMF
