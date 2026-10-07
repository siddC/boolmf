"""The BayesianBooleanMF estimator."""

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
from ._sampler.chain import ChainConfig, LevelPrior, run_chain
from ._sampler.kernels import project_activations
from ._sampler.logsurv import project_activations_ls
from ._sampler.population import MOVE_NAMES as POPULATION_MOVE_NAMES
from ._sampler.population import PopulationConfig, run_population
from ._sampler.splitmerge import MOVE_NAMES
from .diagnostics import ess, rhat
from .matching import match_components
from .utils.validation import matrix_fingerprint, row_seeds, to_binary_int8

DEFAULT_SPLIT_MERGE = 10          # proposals per sweep when split_merge=True


def _transpose_result(r):
    """A chain run on X.T, expressed for X: activations and memberships trade places."""
    r.member_mean, r.activation_mean = r.activation_mean, r.member_mean
    if r.explained is not None:
        r.explained = np.ascontiguousarray(r.explained.T)
        r.predictive = np.ascontiguousarray(r.predictive.T)
    for d in r.draws:
        d["U"], d["Z"] = d["Z"], d["U"]
        if "rho" in d:
            d["pi"], d["rho"] = d["rho"], d["pi"]
    return r


def _is_real(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)

__all__ = ["BayesianBooleanMF", "AnchorComponent", "BetaMixture"]


@dataclass(frozen=True)
class BetaMixture:
    """Two-component Beta mixture prior on activation or membership probabilities.

    Each probability (one per component, per sample or per feature, as the matching
    ``*_level`` says) is drawn from Beta(*component1) when its indicator psi is 1 and from
    Beta(*component0) when it is 0; psi ~ Bernoulli(w) with w ~ Beta(*weight). With a sparse
    and a dense component this is the spike-and-slab prior of Wagala, Samur & Parmigiani
    (2026) on the per-feature membership probabilities.

    Parameters
    ----------
    component1 : tuple of two floats, default=(1.0, 1.0)
        Beta shape parameters (b1, b2) of the probabilities whose indicator is 1.
    component0 : tuple of two floats, default=(1.0, 1.0)
        Beta shape parameters (c1, c2) of the probabilities whose indicator is 0.
    weight : tuple of two floats, default=(1.0, 1.0)
        Beta shape parameters (d1, d2) of the prior on w = P(psi = 1).
    """

    component1: tuple = (1.0, 1.0)
    component0: tuple = (1.0, 1.0)
    weight: tuple = (1.0, 1.0)

    def params(self):
        out = []
        for name in ("component1", "component0", "weight"):
            value = tuple(float(v) for v in getattr(self, name))
            if len(value) != 2 or min(value) <= 0:
                raise ValueError(f"BetaMixture.{name} must be two positive numbers; got "
                                 f"{getattr(self, name)!r}.")
            out.extend(value)
        return tuple(out)


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


def _top_cluster(x, n_clusters, random_state):
    """Entries of a 1-D array in the k-means cluster with the highest center.

    ``sklearn.cluster.KMeans(n_clusters, n_init="auto")`` on the values; the top cluster is set
    to 1 and the others to 0. With fewer distinct values than clusters, one cluster per
    distinct value; a constant array gives no entries.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(x, float).reshape(-1, 1)
    k = min(int(n_clusters), np.unique(x).size)
    if k < 2:
        return np.zeros(x.shape[0], bool)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(x)
    return km.labels_ == int(np.argmax(km.cluster_centers_[:, 0]))


def _redundant_components(components, activations, flags, threshold=0.9):
    """Robust components that the OR model could do without (probabilities > 0.5).

    Two kinds, both usually a component the chain split in two or duplicated:

    * ``("carriers", k, l, jaccard)`` / ``("members", k, l, jaccard)``: components k and l have
      nearly the same carriers (or members). With the same carriers, one component whose
      members are the union of theirs covers the same entries; likewise with the same members.
    * ``("covered", k, -1, fraction)``: at least ``threshold`` of the entries component k covers
      are also covered by other components, so dropping it barely changes the fit.
    """
    idx = np.flatnonzero(np.asarray(flags) == "robust")
    A = np.asarray(activations)[:, idx] > 0.5            # samples x robust components
    B = np.asarray(components)[idx] > 0.5                # robust components x features
    out = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            for side, M in (("carriers", A.T), ("members", B)):
                inter = np.count_nonzero(M[a] & M[b])
                union = np.count_nonzero(M[a] | M[b])
                if union and inter / union >= threshold:
                    out.append((side, int(idx[a]), int(idx[b]), inter / union))
                    break
    if len(idx) > 1:
        count = A.astype(np.int32) @ B.astype(np.int32)   # components covering each entry
        for a in range(len(idx)):
            sub = count[np.ix_(A[:, a], B[a])]
            if sub.size and np.mean(sub >= 2) >= threshold:
                out.append(("covered", int(idx[a]), -1, float(np.mean(sub >= 2))))
    return out


def _bayes_fdr_threshold(p, fdr):
    """Smallest threshold whose calls have expected FDR = mean(1 - p) <= fdr."""
    s = np.sort(np.asarray(p, float).ravel())[::-1]
    if s.size == 0:
        return 1.0
    cum = np.cumsum(1.0 - s) / np.arange(1, s.size + 1)
    ok = np.flatnonzero(cum <= fdr)
    return float(s[ok[-1]]) if ok.size else 1.0


class BayesianBooleanMF(TransformerMixin, BaseEstimator):
    """Bayesian nonparametric Boolean matrix factorization.

    Models a binary matrix X (samples x features) as the union of latent components: each
    component has binary members (features) and is active or inactive in each sample, and a
    feature is present when an active component containing it transmits it, or through
    background presence. With ``n_components=None`` the number of components is given an
    Indian buffet process prior and learned from the data. The posterior is sampled with
    several independent Gibbs chains whose components are matched with the Hungarian
    algorithm and pooled. For the standard (non-Bayesian) algorithms see ``BooleanMF``.

    Parameters
    ----------
    n_components : int or None, default=None
        None learns the number of components (Indian buffet process); an int fixes it.
    max_components : int or "auto", default="auto"
        Number of component slots (truncation of the Indian buffet process).
        ``"auto"`` uses ``max(1, min(n_samples, n_features) // 2)``, or
        ``min(n_samples, n_features) + 2 * max_births`` with ``births`` other than
        ``"slots"``. A warning is raised when more than 80% of the slots are in use (or, with
        the collapsed sampler, when a birth found no free slot), since the cap then limits the
        rank.
    likelihood : {"or_flip", "noisy_or"}, default="or_flip"
        ``"or_flip"``: P(x = 1) = detection if c >= 1, else background, where c is the number
        of active components containing the feature. ``"noisy_or"``: P(x = 1) =
        1 - (1 - background) (1 - detection)^c, so a feature in two active components is more
        likely observed. The two agree whenever c <= 1. With ``"noisy_or"`` a component whose
        members are split across two components with the same carriers is a local mode that
        one-at-a-time updates rarely leave; split-merge moves (planned for v0.2) address this,
        so ``"or_flip"`` is the default for now.
    likelihood_power : float in (0, 1], default=1.0
        The power zeta of the likelihood in the target, likelihood^zeta x prior. 1 is the
        usual posterior. Below 1 it is a coarsened (power) posterior (Miller & Dunson 2019,
        Robust Bayesian inference via coarsening, JASA 114: 1113-1125): every observed entry
        counts as zeta of an entry, so structure must be supported by more data before the
        model adds a component for it. Real data are never exactly Boolean, and with the full
        likelihood the number of components keeps growing with the data; coarsening sets how
        much misfit is tolerated. Miller & Dunson write zeta = a / (a + N) for N observed
        entries and choose a from where the fit stops improving as a grows. The reported
        log-likelihood (``log_likelihood_trace_``) and the predictive probabilities are
        those of the model itself, not tempered. Not available with
        ``detection_effects=("component",)``.
    anchor_components : list of AnchorComponent or None, default=None
        Components active in every sample, for example a core-genome component.
    binarize : float or None, default=None
        None requires 0/1 input (NaN = missing). A float maps values above it to 1.
    detection_effects, background_effects : tuple of str, default=()
        Levels at which the rate varies. ``("sample",)`` gives every sample its own rate,
        a_i = sigmoid(y_i) with y_i ~ Normal(mu, sigma^2) on the logit scale; the spread sigma
        is learned (half-Cauchy(0, 1) prior) and sigmoid(mu) is the population rate. Under
        ``or_flip`` each sample's detection rate stays above its background rate.
        ``detection_effects=("component",)`` gives every component its own detection rate
        (logit-normal around the population rate, learned spread); it requires
        ``likelihood="noisy_or"``, where each active component delivers each of its features
        independently. With both levels, logit lambda_ik = y_i + g_k. Per-feature rates are
        planned for v0.3.
    detection_prior, background_prior : scipy.stats frozen distribution, float or None
        Priors on the two rates (on the population rate when the rate varies by sample). None
        is Beta(1, 1). A float in (0, 1) fixes the rate. Any distribution with support in
        [0, 1] is accepted.
    membership_prior : scipy.stats.beta frozen distribution, BetaMixture, float, "empirical" \
or None
        Prior on the probability that a feature belongs to a component. A Beta distribution
        (None is Beta(1, 1)) puts a random rate at ``membership_level``; a ``BetaMixture``
        (fixed ``n_components`` only) draws each rate from one of two Beta distributions, as
        in Wagala et al. (2026); a float fixes the rate; ``"empirical"`` fixes it from the data
        density as in Rukat et al. (2017), p = sqrt(1 - (1 - density)^(1 / n_components)).
    membership_level : {"component", "shared", "feature"}, default="component"
        Where a Beta membership rate lives: one per component, one shared by all components,
        or one per feature.
    activation_prior : scipy.stats.beta frozen distribution, BetaMixture, float, \
"empirical" or None
        Prior on the probability that a component is active in a sample, used when
        ``n_components`` is set (otherwise the Indian buffet process is the prior). None is
        Beta(1, 1); a ``BetaMixture``, a float or ``"empirical"`` as for ``membership_prior``.
    activation_level : {"component", "shared", "sample"}, default="component"
        Where a Beta activation rate lives.
    alpha_prior : scipy.stats.gamma frozen distribution, float or None, default=None
        Prior on the Indian buffet process concentration. None is Gamma(1, 1); a float fixes
        alpha.
    tied_rates : bool, default=False
        ``or_flip`` only: background = 1 - detection, the symmetric flip noise of the
        OrMachine (Rukat et al. 2017).
    rate_estimation : {"bayes", "mle"}, default="bayes"
        ``"bayes"`` samples the rates from their posterior. ``"mle"`` (``or_flip`` only) sets
        them to their maximum-likelihood values after every sweep, as the OrMachine does.
    update : {"gibbs", "metropolised"}, default="gibbs"
        How each binary variable is updated: drawn from its conditional, or flipped with
        probability min(1, p(flipped) / p(current)) (Metropolised Gibbs, Liu 1996; the
        OrMachine's sampler).
    update_order : {"memberships_first", "activations_first"}, default="memberships_first"
        Which factor matrix each sweep updates first.
    births : {"slots", "enumerate", "metropolis"}, default="slots"
        How components are born and die under the Indian buffet process. ``"slots"``: a
        truncated pool of ``max_components`` slots whose probabilities are sampled, with empty
        slots switched on by Gibbs updates. ``"enumerate"``: the collapsed sampler of Wood,
        Griffiths & Ghahramani (2006): each row keeps existing components with probability
        m / N and draws its number of new components from the exact conditional (up to
        ``max_births``) with their memberships summed out. ``"metropolis"``: the same
        collapsed sampler with the births of Meeds et al. (2007), a Poisson(alpha / N) number
        of new components replacing the row's own ones, accepted by Metropolis-Hastings. Both
        need a shared (or fixed) probability on the other side (``membership_level="shared"``
        or a float ``membership_prior``; with ``ibp_side="features"``, the activation side)
        and ``split_merge=False``.
    max_births : int, default=10
        Most new components one row can add in one step (Wood et al. truncate at 10).
    birth_members : {"exact", "gibbs"}, default="exact"
        With ``births="enumerate"``, how the other side of the new components is drawn.
        ``"exact"``: from its joint conditional given the row. ``"gibbs"``: starting from
        zero, one Gibbs pass over the new components one at a time, as written in Wood et al.
        (2006) and Rukat & Yau (2019). The two agree when one component is born; with several,
        the pass is not a draw from the joint conditional, so the chain does not target the
        exact posterior, but it gives the published transients: the first new component takes
        most of the row's unexplained entries instead of a random share.
    ibp_side : {"samples", "features"}, default="samples"
        Which side carries the Indian buffet process. Wood et al. (2006) put it on the
        observed variables (features), with sample activations independent Bernoulli(p):
        ``ibp_side="features"`` with ``activation_level="shared"`` or a float
        ``activation_prior``.
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
    population_moves : int, default=0
        Sweeps between rounds of population moves; 0 runs the chains independently.
        Otherwise the chains run in lockstep in one process (``n_jobs`` is ignored) and every
        ``population_moves`` sweeps propose moves that use the other chains' components: a
        component of another chain copied into an empty slot (transplant) or a component that
        another chain also has removed (delete), and every component inside a region of the
        data swapped between two chains (crossover; regions are the clusters of a fixed
        average-linkage clustering of the samples and of the features, Jaccard distance).
        Each move is a Metropolis-Hastings step whose proposal depends on another chain's
        current state, so every chain still samples its own posterior (evolutionary Monte
        Carlo, Liang & Wong 2000; Jasra, Stephens & Holmes 2007). Independent chains of a
        Boolean factorization often each find part of the structure; these moves let a chain
        adopt what another found when it raises its posterior. Needs ``births="slots"``,
        ``n_chains >= 2``, a Beta rate per component on both sides (as split-merge does) and
        rates shared by components.
    population_params : dict or None, default=None
        ``n_transplant`` (transplant or delete proposals per chain and round, default 20),
        ``n_crossover`` (crossover proposals per round, default 10), ``region_min_size``
        (default 3), ``region_max_fraction`` (largest region as a fraction of the samples or
        features, default 0.5) and ``max_region_items`` (no feature regions with more features
        than this, default 20000).
    rhat_threshold : float, default=1.01
        Convergence cut-off. A warning is raised when the cross-chain R-hat of the detection or
        background rate exceeds ``max(1.05, rhat_threshold)``.
    min_ess : float, default=400
        A warning is raised when the pooled effective sample size of the detection or
        background rate is below this value (checked when ``n_chains * n_draws`` is at least
        twice ``min_ess``). The log-likelihood and component count are reported in ``ess_``
        but do not trigger warnings: they mix slowly because short-lived components that
        absorb noise come and go, which leaves robust components unaffected.
    init : {"random", "uniform", "empty", "nmf", "asso"} or tuple of (members, activations)
        Starting state for the chains (default ``"random"``: a few sparse random components).
        ``"uniform"`` sets every entry of every component to 0 or 1 with probability 1/2;
        ``"empty"`` starts with no components. ``"nmf"`` fits
        ``sklearn.decomposition.NMF(**init_params)`` (NNDSVDa start unless ``init_params``
        sets ``init``) and binarizes each component and each column of its activations by
        k-means on the values (``KMeans(n_clusters=3, n_init="auto")``): the cluster with the
        highest center becomes 1, the other two 0. This is the binarization used for NMF
        phylons; it favors precision over recall. ``init_params["binarize_clusters"]`` sets
        the number of clusters (2 splits each component in two). ``"asso"``
        starts every chain from the Asso factorization (``BooleanMF(algorithm="asso")``,
        threshold 0.5 and unit weights unless ``init_params`` sets ``threshold``,
        ``positive_weight`` or ``negative_weight``; missing entries count as 0). A tuple gives
        members (n_init, n_features) and activations (n_samples, n_init) directly; a list of
        ``n_chains`` such tuples gives each chain its own start.
    init_params : dict or None, default=None
        Keyword arguments for ``sklearn.decomposition.NMF`` when ``init="nmf"`` (plus
        ``binarize_clusters``, see ``init``), or for Asso when ``init="asso"``.
    robustness_threshold : float, default=0.5
        Fraction of chains a component must appear in to be flagged robust.
    min_support : float, default=3
        Components with fewer expected members or active samples are flagged low-support.
    store_draws : bool or int, default=True
        Keep bit-packed binary draws (needed by ``transform`` and ``get_draws``). An int keeps
        at most that many per chain, evenly spaced over the kept draws.
    checkpoint_dir : str, path or None, default=None
        Directory where each chain saves its complete state every ``checkpoint_every``
        sweeps (one file per chain, written atomically). Fitting again with the same data,
        settings and an int ``random_state`` resumes every chain from its last checkpoint, so
        a long fit survives an interrupted process; the result is the same as an
        uninterrupted fit. A chain's file is removed when it finishes. A checkpoint written for
        other data or settings raises ``ValueError``.
    checkpoint_every : int, default=200
        Sweeps between checkpoints.
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
    redundant_components_ : list of tuple
        Signs that robust components are redundant, as ``(kind, k, l, score)``:
        ``("carriers", k, l, jaccard)`` or ``("members", k, l, jaccard)`` when components k and
        l have carriers (or members, probability above 0.5) with Jaccard similarity of at least
        0.9, and ``("covered", k, -1, fraction)`` when at least 90% of the entries component k
        covers are covered by other components. Either usually marks a component that the
        sampler split in two or duplicated; a warning is raised when the list is not empty.
    prevalence_ : ndarray
        Mean activation of each component across samples.
    detection_rate_, background_rate_ : float
        Posterior mean rates; the population rate when the rate varies by sample.
    detection_rate_per_sample_, background_rate_per_sample_ : ndarray of shape (n_samples,)
        Posterior mean rate of each training sample; only when ``"sample"`` is in the matching
        ``*_effects``. ``*_rate_per_sample_interval_`` holds 95% credible intervals, shape
        (n_samples, 2), and ``detection_spread_`` / ``background_spread_`` the posterior mean
        spread of the logit rates.
    detection_rate_per_component_ : ndarray of shape (n_components_total,)
        Posterior mean detection rate of each component (at the population level); only with
        ``detection_effects`` containing ``"component"``. ``*_interval_`` holds 95% credible
        intervals and ``detection_component_spread_`` the spread of the logit rates.
    alpha_ : float or None
        Posterior mean of the Indian buffet process concentration.
    map_components_, map_activations_ : ndarray of uint8 or None
        The kept draw with the highest unnormalized log posterior over all good chains (the
        point estimate of Wagala et al. 2026), its components in the order of
        ``components_``: shapes (n_components_total, n_features) and (n_samples,
        n_components_total). Only with a fixed ``n_components``, no anchors and global rates;
        otherwise None. ``map_log_posterior_``, ``map_detection_rate_`` and
        ``map_background_rate_`` belong to the same draw.
    n_components_draws_ : ndarray of shape (n_good_chains, n_draws)
        Posterior draws of the number of active (non-anchor) components.
    rhat_, ess_ : dict
        Cross-chain R-hat and effective sample size of the monitored scalars.
    split_merge_acceptance_ : dict
        Share of proposals accepted per move type (split, merge, reallocate, factor,
        unfactor), pooled over chains.
    population_acceptance_ : dict or None
        With ``population_moves``: proposals and the share accepted per move type
        (transplant, delete, crossover_samples, crossover_features), as
        ``{name: (n_proposed, share_accepted)}``. None otherwise.
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
        likelihood_power=1.0,
        anchor_components=None,
        binarize=None,
        detection_effects=(),
        background_effects=(),
        detection_prior=None,
        background_prior=None,
        membership_prior=None,
        membership_level="component",
        activation_prior=None,
        activation_level="component",
        alpha_prior=None,
        tied_rates=False,
        rate_estimation="bayes",
        update="gibbs",
        update_order="memberships_first",
        births="slots",
        max_births=10,
        birth_members="exact",
        ibp_side="samples",
        n_chains=20,
        max_sweeps=10000,
        burn_in="auto",
        n_draws=100,
        thin="auto",
        split_merge=True,
        population_moves=0,
        population_params=None,
        rhat_threshold=1.01,
        min_ess=400,
        init="random",
        init_params=None,
        robustness_threshold=0.5,
        min_support=3,
        store_draws=True,
        checkpoint_dir=None,
        checkpoint_every=200,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.max_components = max_components
        self.likelihood = likelihood
        self.likelihood_power = likelihood_power
        self.anchor_components = anchor_components
        self.binarize = binarize
        self.detection_effects = detection_effects
        self.background_effects = background_effects
        self.detection_prior = detection_prior
        self.background_prior = background_prior
        self.membership_prior = membership_prior
        self.membership_level = membership_level
        self.activation_prior = activation_prior
        self.activation_level = activation_level
        self.alpha_prior = alpha_prior
        self.tied_rates = tied_rates
        self.rate_estimation = rate_estimation
        self.update = update
        self.update_order = update_order
        self.births = births
        self.max_births = max_births
        self.birth_members = birth_members
        self.ibp_side = ibp_side
        self.n_chains = n_chains
        self.max_sweeps = max_sweeps
        self.burn_in = burn_in
        self.n_draws = n_draws
        self.thin = thin
        self.split_merge = split_merge
        self.population_moves = population_moves
        self.population_params = population_params
        self.rhat_threshold = rhat_threshold
        self.min_ess = min_ess
        self.init = init
        self.init_params = init_params
        self.robustness_threshold = robustness_threshold
        self.min_support = min_support
        self.store_draws = store_draws
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ sklearn plumbing
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        return tags

    def _level_prior(self, name, level, density, n_components):
        """LevelPrior from a ``*_prior`` parameter: Beta (or None), float, or "empirical"."""
        value = getattr(self, name)
        lev = {"component": "component", "shared": "shared"}.get(level, "row")
        if isinstance(value, str):
            if value != "empirical":
                raise ValueError(f"{name} must be a Beta distribution, a float, 'empirical' "
                                 f"or None; got {value!r}.")
            L = max(1, n_components)
            p = float(np.sqrt(1.0 - (1.0 - min(density, 1 - 1e-12)) ** (1.0 / L)))
            return LevelPrior("fixed", "shared", value=float(np.clip(p, 1e-6, 1 - 1e-6)))
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} fixed at {value}; a fixed rate must lie in (0, 1).")
            return LevelPrior("fixed", "shared", value=float(value))
        if isinstance(value, BetaMixture):
            return LevelPrior("mixture", lev, mix=value.params())
        a, b = beta_params_or_raise(value, name)
        return LevelPrior("beta", lev, a, b)

    def _levels(self, name):
        value = getattr(self, name)
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        try:
            return tuple(value)
        except TypeError:
            raise ValueError(f"{name} must be a tuple of level names; got {value!r}.") from None

    def _n_split_merge(self):
        if isinstance(self.split_merge, (bool, np.bool_)):
            return DEFAULT_SPLIT_MERGE if self.split_merge else 0
        return int(self.split_merge)

    def _population_config(self):
        params = dict(self.population_params or {})
        pcfg = PopulationConfig(every=int(self.population_moves))
        for name in ("n_transplant", "n_crossover", "region_min_size", "max_region_items"):
            if name in params:
                v = params.pop(name)
                if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < 0:
                    raise ValueError(f"population_params[{name!r}] must be an int >= 0; "
                                     f"got {v!r}.")
                setattr(pcfg, name, int(v))
        if "region_max_fraction" in params:
            v = params.pop("region_max_fraction")
            if not isinstance(v, numbers.Real) or isinstance(v, bool) or not 0 < v <= 1:
                raise ValueError("population_params['region_max_fraction'] must be in (0, 1]; "
                                 f"got {v!r}.")
            pcfg.region_max_fraction = float(v)
        if params:
            raise ValueError(f"population_params does not take {sorted(params)}.")
        return pcfg

    def _check_population(self):
        v = self.population_moves
        if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < 0:
            raise ValueError(f"population_moves must be an int >= 0; got {v!r}.")
        if self.population_params is not None and not isinstance(self.population_params, dict):
            raise ValueError("population_params must be a dict or None.")
        self._population_config()
        if v == 0:
            return
        if self.n_chains < 2:
            raise ValueError("population_moves needs n_chains >= 2.")
        if self.births != "slots":
            raise ValueError("population_moves needs births='slots'.")
        if "component" in self._levels("detection_effects"):
            raise NotImplementedError(
                "population_moves with detection_effects=('component',) is not implemented.")
        for name, level in (("membership", self.membership_level),
                            ("activation", self.activation_level)):
            pri = getattr(self, f"{name}_prior")
            if level != "component" or isinstance(pri, BetaMixture) \
                    or isinstance(pri, (numbers.Real, str)) and not isinstance(pri, bool):
                raise ValueError(
                    f"population_moves needs a Beta {name} rate per component; got "
                    f"{name}_level={level!r} and {name}_prior={pri!r}.")

    def _check_params(self):
        def _int(name, lo):
            v = getattr(self, name)
            if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < lo:
                raise ValueError(f"{name} must be an int >= {lo}; got {v!r}.")

        if self.n_components is not None:
            _int("n_components", 1)
        if self.max_components != "auto":
            _int("max_components", 1)
        if not (isinstance(self.likelihood_power, numbers.Real)
                and not isinstance(self.likelihood_power, bool)
                and 0.0 < self.likelihood_power <= 1.0):
            raise ValueError(f"likelihood_power must be in (0, 1]; got {self.likelihood_power!r}.")
        if self.likelihood_power != 1.0 and "component" in self._levels("detection_effects"):
            raise NotImplementedError(
                "likelihood_power < 1 with detection_effects=('component',) is not implemented.")
        for name, allowed in (("detection_effects", ("sample", "component")),
                              ("background_effects", ("sample",))):
            levels = self._levels(name)
            for lev in levels:
                if lev == "feature":
                    raise NotImplementedError(f"{name}: per-feature rates are planned for v0.3.")
                if lev not in allowed:
                    raise ValueError(f"{name} must contain only {allowed}; got {lev!r}.")
            if name == "detection_effects" and "component" in levels \
                    and self.likelihood != "noisy_or":
                raise ValueError(
                    "detection_effects=('component',) requires likelihood='noisy_or'. Under "
                    "'or_flip' an entry covered by several components is present with a single "
                    "detection rate, so no component's own rate can be identified; under "
                    "'noisy_or' each active component delivers its features independently, "
                    "with its own rate.")
            prior = self.detection_prior if name.startswith("detection") else self.background_prior
            if levels and isinstance(prior, numbers.Real) and not isinstance(prior, bool):
                raise ValueError(f"{name} needs a free rate; the matching prior fixes it.")
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
        if isinstance(self.init, list):
            if len(self.init) != self.n_chains or not all(
                    isinstance(x, tuple) and len(x) == 2 for x in self.init):
                raise ValueError("a list init needs one (members, activations) tuple per chain "
                                 f"({self.n_chains}); got {len(self.init)} item(s).")
        elif not (isinstance(self.init, tuple)
                  or self.init in ("random", "uniform", "empty", "nmf", "asso")):
            raise ValueError("init must be 'random', 'uniform', 'empty', 'nmf', 'asso', a "
                             "(members, activations) tuple or a list of them (one per chain); "
                             f"got {self.init!r}.")
        self._check_population()
        for name, allowed in (("membership_level", ("component", "shared", "feature")),
                              ("activation_level", ("component", "shared", "sample")),
                              ("rate_estimation", ("bayes", "mle")),
                              ("update", ("gibbs", "metropolised")),
                              ("update_order", ("memberships_first", "activations_first"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {getattr(self, name)!r}.")
        for name, allowed in (("births", ("slots", "enumerate", "metropolis")),
                              ("birth_members", ("exact", "gibbs")),
                              ("ibp_side", ("samples", "features"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {getattr(self, name)!r}.")
        _int("max_births", 1)
        ibp_features = self.ibp_side == "features"
        if ibp_features and self.n_components is not None:
            raise ValueError("ibp_side='features' needs n_components=None (the IBP).")
        if ibp_features and (self.anchor_components or self._levels("detection_effects")
                             or self._levels("background_effects")):
            raise ValueError("ibp_side='features' does not combine with anchor components or "
                             "rate effects.")
        side = "activation" if ibp_features else "membership"
        if self.n_components is None and not ibp_features and self.activation_prior is not None:
            raise ValueError("activation_prior applies with a fixed n_components; without it "
                             "the Indian buffet process is the prior on activations.")
        if self.birth_members != "exact" and self.births != "enumerate":
            raise ValueError("birth_members applies with births='enumerate'.")
        if self.births != "slots":
            if self.n_components is not None:
                raise ValueError(f"births={self.births!r} is an IBP sampler; it needs "
                                 "n_components=None.")
            if self._n_split_merge() > 0:
                raise ValueError(f"births={self.births!r} runs without split-merge moves; set "
                                 "split_merge=False.")
            other = getattr(self, f"{side}_prior")
            if not (_is_real(other) or other == "empirical"
                    or getattr(self, f"{side}_level") == "shared"):
                raise ValueError(
                    f"births={self.births!r} needs one {side} probability shared by all "
                    f"components: set {side}_level='shared' or a float {side}_prior.")
            if self._levels("detection_effects") or self._levels("background_effects") \
                    or self.anchor_components:
                raise ValueError(f"births={self.births!r} does not combine with rate effects "
                                 "or anchor components.")
        if self.tied_rates and self.likelihood != "or_flip":
            raise ValueError("tied_rates needs likelihood='or_flip' (symmetric flip noise).")
        if self.rate_estimation == "mle" and self.likelihood != "or_flip":
            raise ValueError("rate_estimation='mle' is available for likelihood='or_flip'.")
        effects = self._levels("detection_effects") + self._levels("background_effects")
        if effects and (self.tied_rates or self.rate_estimation == "mle"):
            raise ValueError("tied_rates and rate_estimation='mle' need global rates "
                             "(no detection_effects or background_effects).")
        if "component" in self._levels("detection_effects") and self.update != "gibbs":
            raise ValueError("update='metropolised' is not available with component rates.")
        if self._n_split_merge() > 0:
            for name, level in (("membership", self.membership_level),
                                ("activation", self.activation_level)):
                pri = getattr(self, f"{name}_prior")
                if level != "component" or isinstance(pri, BetaMixture) \
                        or isinstance(pri, (numbers.Real, str)) and not isinstance(pri, bool):
                    raise ValueError(
                        f"split_merge needs a Beta {name} rate per component; set "
                        f"split_merge=False to use {name}_level={level!r} or a fixed rate.")
        for name in ("activation_prior", "membership_prior"):
            if isinstance(getattr(self, name), BetaMixture):
                if self.n_components is None:
                    raise ValueError(f"a BetaMixture {name} needs a fixed n_components.")
                getattr(self, name).params()
        _int("checkpoint_every", 1)
        if self.checkpoint_dir is not None:
            if self.random_state is None or not isinstance(self.random_state, numbers.Integral):
                raise ValueError("checkpoint_dir needs an int random_state, so that a refit "
                                 "can find each chain's checkpoint.")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not (isinstance(self.store_draws, (bool, np.bool_)) or (
                isinstance(self.store_draws, numbers.Integral) and self.store_draws >= 1)):
            raise ValueError("store_draws must be a bool or a positive int.")
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
            if self.max_components != "auto":
                n_free = int(self.max_components)
            elif self.births == "slots":
                n_free = max(1, min(n, F) // 2)
            else:                       # the collapsed sampler has no truncation of its own
                n_free = min(n, F) + 2 * int(self.max_births)
            nonparametric = True
        n_slots = na + n_free
        n_init = n_free if not nonparametric else int(
            min(n_free, max(2, np.ceil(2 * np.sqrt(min(n, F))))))

        rs = check_random_state(self.random_state)
        base_seed = int(rs.randint(0, 2**31 - 1))
        ss = np.random.SeedSequence(base_seed)
        chain_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(self.n_chains)]
        self._transform_seed_ = int(np.random.SeedSequence(base_seed + 1).generate_state(1)[0])

        flip = self.ibp_side == "features"
        V_fit = np.ascontiguousarray(V.T) if flip else V
        init = self._initial_state(V, n_free)
        inits = init if isinstance(init, list) else [init] * self.n_chains
        if flip:
            inits = [(x[1].T, x[0].T) if isinstance(x, tuple) else x for x in inits]
        population = int(self.population_moves) > 0
        pcfg = self._population_config() if population else None
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, min(n_jobs, self.n_chains))
        n_threads = max(1, (os.cpu_count() or 1) // max(1, n_workers)) if n_workers > 1 else 0

        density = float((V == 1).sum() / max((V >= 0).sum(), 1))
        if flip:        # fit the transpose: the IBP sits on its rows, the samples' prior on U
            mem_spec = self._level_prior("activation_prior", self.activation_level, density,
                                         n_free)
            act_spec = None
        else:
            mem_spec = self._level_prior("membership_prior", self.membership_level, density,
                                         n_free)
            act_spec = None if nonparametric else self._level_prior(
                "activation_prior", self.activation_level, density, n_free)
        mem_prior_spec_ab = (mem_spec.a, mem_spec.b) if mem_spec.kind == "beta" else (1.0, 1.0)
        cfg = ChainConfig(
            likelihood=self.likelihood,
            n_slots=n_slots,
            anchor_members=anchors,
            anchor_learned=anchor_learned,
            nonparametric=nonparametric,
            prior_a=RatePrior.from_param(self.detection_prior, "detection_prior"),
            prior_b=RatePrior.from_param(self.background_prior, "background_prior"),
            membership_ab=mem_prior_spec_ab,
            alpha_prior=PositivePrior.from_param(self.alpha_prior, "alpha_prior"),
            burn_in=self.burn_in if self.burn_in == "auto" else int(self.burn_in),
            max_sweeps=int(max(self.max_sweeps, self.n_draws + 1)),
            n_draws=int(self.n_draws),
            thin=self.thin if self.thin == "auto" else int(self.thin),
            rhat_threshold=float(self.rhat_threshold),
            n_init=n_init,
            store_draws=self.store_draws if isinstance(self.store_draws, (bool, np.bool_))
            else int(self.store_draws),
            checkpoint_dir=None if self.checkpoint_dir is None else os.fspath(self.checkpoint_dir),
            checkpoint_every=int(self.checkpoint_every),
            data_key=matrix_fingerprint(V_fit) + repr(
                [self._init_key(x) for x in inits] if isinstance(init, list)
                else self._init_key(inits[0])),
            population=repr(vars(pcfg)) if population else "",
            track_map=(not nonparametric and not na
                       and not (self._levels("detection_effects")
                                or self._levels("background_effects"))),
            store_entries=True,
            n_threads=0 if population else n_threads,
            min_support=int(np.ceil(self.min_support)),
            burn_rhat=max(1.05, float(self.rhat_threshold)),
            n_split_merge=self._n_split_merge(),
            detection_effects=self._levels("detection_effects"),
            background_effects=self._levels("background_effects"),
            likelihood_power=float(self.likelihood_power),
            activation_prior=act_spec,
            membership_prior=mem_spec,
            tied_rates=bool(self.tied_rates),
            rate_estimation=self.rate_estimation,
            metropolis=self.update == "metropolised",
            activations_first=self.update_order == "activations_first",
            init_kind=self.init if isinstance(self.init, str) else "random",
            births=self.births,
            max_new=int(self.max_births),
            exact_birth_members=self.birth_members == "exact",
            verbose=self.verbose,
        )
        self.population_acceptance_ = None
        if population:
            pop_seed = int(np.random.SeedSequence(base_seed + 2).generate_state(1)[0])
            results, pstats = run_population(V_fit, cfg, chain_seeds, inits, pcfg, pop_seed)
            self.population_acceptance_ = {
                name: (int(pstats[0, t]), float(pstats[1, t] / pstats[0, t]) if pstats[0, t]
                       else float("nan"))
                for t, name in enumerate(POPULATION_MOVE_NAMES)}
        else:
            results = Parallel(n_jobs=n_workers, verbose=0)(
                delayed(run_chain)(V_fit, cfg, s, x) for s, x in zip(chain_seeds, inits)
            )
        if flip:
            results = [_transpose_result(r) for r in results]
        self._postprocess(V, results, na, nonparametric, n_free)
        return self

    @staticmethod
    def _init_key(init):
        if isinstance(init, tuple):
            return tuple(matrix_fingerprint(np.ascontiguousarray(x, dtype=np.int8)) for x in init)
        return init

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

    def _initial_state(self, V, n_free, init=None):
        init = self.init if init is None else init
        if isinstance(init, list):
            return [self._initial_state(V, n_free, x) for x in init]
        if isinstance(init, tuple):
            members, activations = (np.asarray(x) for x in init)
            if members.ndim != 2 or members.shape[1] != V.shape[1]:
                raise ValueError("init members must have shape (n_init, n_features).")
            if activations.shape != (V.shape[0], members.shape[0]):
                raise ValueError("init activations must have shape (n_samples, n_init).")
            if members.shape[0] > n_free:
                warnings.warn("init has more components than slots; extra ones are dropped.",
                              stacklevel=3)
            return (members.astype(bool), activations.astype(bool))
        if self.init == "asso":
            from ._algorithms import asso

            params = dict(self.init_params or {})
            k = int(self.n_components) if self.n_components is not None else int(
                min(n_free, max(2, np.sqrt(min(V.shape)))))
            W, H = asso(V == 1, params.pop("n_components", k), params.pop("threshold", 0.5),
                        params.pop("positive_weight", 1.0), params.pop("negative_weight", 1.0))
            if params:
                raise ValueError(f"init='asso' does not take {sorted(params)}.")
            return (H, W)
        if self.init == "nmf":
            from sklearn.decomposition import NMF

            params = dict(self.init_params or {})
            n_clusters = params.pop("binarize_clusters", 3)
            if not (isinstance(n_clusters, numbers.Integral) and n_clusters >= 2):
                raise ValueError(f"init_params['binarize_clusters'] must be an int >= 2; got "
                                 f"{n_clusters!r}.")
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
            seed = params.get("random_state")
            members = np.stack([_top_cluster(H[k], n_clusters, seed) for k in range(H.shape[0])])
            acts = np.stack([_top_cluster(W[:, k], n_clusters, seed) for k in range(W.shape[1])],
                            axis=1)
            return (members, acts)
        return "random"

    def _set_map(self, results, good, n, F):
        """The draw with the highest log posterior over all good chains (finite models with
        global rates), with its components put in the order of ``components_``."""
        for attr in ("map_components_", "map_activations_", "map_log_posterior_",
                     "map_detection_rate_", "map_background_rate_"):
            setattr(self, attr, None)
        states = [(int(c), results[c].map_state) for c in good
                  if results[c].map_state is not None]
        if not states:
            return
        c, (lp, Zm, Um, a, b) = max(states, key=lambda t: t[1][0])
        Kt = self.components_.shape[0]
        H = np.zeros((Kt, F), np.uint8)
        W = np.zeros((n, Kt), np.uint8)
        for slot, k in self._slot_map_[c].items():
            H[k] = Um[:, slot]
            W[:, k] = Zm[:, slot]
        self.map_components_, self.map_activations_ = H, W
        self.map_log_posterior_ = float(lp)
        self.map_detection_rate_, self.map_background_rate_ = float(a), float(b)

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
            if nonparametric:
                keep = np.flatnonzero((m.sum(0) >= 1) & (a.sum(0) >= 1))
            else:                   # a fixed number of components: every slot in use counts
                keep = np.flatnonzero((r.member_mean[:, na:].sum(0) > 0)
                                      & (r.activation_mean[:, na:].sum(0) > 0))
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
        self._set_map(results, good, n, F)

        rates = np.concatenate([results[c].draw_rates for c in good])
        self.detection_rate_ = float(rates[:, 0].mean())
        self.background_rate_ = float(rates[:, 1].mean())
        for kind in ("detection", "background"):
            for attr in (f"{kind}_rate_per_sample_", f"{kind}_rate_per_sample_interval_",
                         f"{kind}_spread_"):
                if hasattr(self, attr):
                    delattr(self, attr)                    # from an earlier fit
            per = [results[c].draw_sample_rates.get(kind) for c in good]
            if per[0] is None:
                continue
            per = np.concatenate(per)                          # (draws, n_samples)
            setattr(self, f"{kind}_rate_per_sample_", per.mean(0).astype(float))
            setattr(self, f"{kind}_rate_per_sample_interval_",
                    np.quantile(per, [0.025, 0.975], axis=0).T.astype(float))
            col = 0 if kind == "detection" else 1
            spread = np.concatenate([results[c].draw_spread[:, col] for c in good])
            setattr(self, f"{kind}_spread_", float(spread.mean()))
        for attr in ("detection_rate_per_component_", "detection_rate_per_component_interval_",
                     "detection_component_spread_"):
            if hasattr(self, attr):
                delattr(self, attr)
        if "component" in self._levels("detection_effects"):
            lam = [[] for _ in range(len(flags))]
            for c in good:
                smap = self._slot_map_[c]
                for slots, values in results[c].draw_slot_rates:
                    for slot, value in zip(slots, values):
                        k = smap.get(int(slot))
                        if k is not None:
                            lam[k].append(value)
            self.detection_rate_per_component_ = np.array(
                [np.mean(v) if v else np.nan for v in lam])
            self.detection_rate_per_component_interval_ = np.array(
                [np.quantile(v, [0.025, 0.975]) if v else [np.nan, np.nan] for v in lam])
            self.detection_component_spread_ = float(np.mean(np.concatenate(
                [results[c].draw_spread[:, 2] for c in good])))
        self.alpha_ = float(np.nanmean(np.concatenate([results[c].draw_alpha for c in good]))) \
            if nonparametric else None
        if self.births != "slots" and any(results[c].births[2] > 0 for c in good):
            warnings.warn(
                "births were cut short because every component slot was in use; increase "
                "max_components.", stacklevel=3)
        elif nonparametric and np.median(self.n_components_draws_) > 0.8 * n_free:
            warnings.warn(
                "more than 80% of component slots are in use; increase max_components.",
                stacklevel=3)

        self.redundant_components_ = _redundant_components(
            self.components_, self.activations_, self.component_flags_)
        if self.redundant_components_:
            kind, k, l, _ = self.redundant_components_[0]
            first = (f"component {k} is almost entirely covered by the others"
                     if kind == "covered" else f"components {k} and {l} share their {kind}")
            warnings.warn(
                f"{len(self.redundant_components_)} sign(s) of redundant components (first: "
                f"{first}; see redundant_components_). Under the OR model such components add "
                "almost nothing to the fit, so they usually mark one component split in two or "
                "duplicated, a mode that one-at-a-time updates leave slowly. Split-merge moves "
                "(births='slots', split_merge=True) or more chains usually remove it.",
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
                        "lam": d.get("lam"),
                    })
        self._n_train_samples_ = n

    # ------------------------------------------------------------------ inference outputs
    def fit_transform(self, X, y=None, mask=None):
        """Fit, then ``transform`` the training samples (as scikit-learn expects).

        ``activations_`` holds the activation probabilities from the chains themselves, which
        sample activations and components jointly; ``transform`` holds the components fixed
        at stored draws, so the two agree closely but not exactly.
        """
        return self.fit(X, y, mask=mask).transform(X, mask=mask)

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
            pi = np.clip(d["pi"], 1e-12, 1 - 1e-12)
            logit_pi = np.log(pi) - np.log1p(-pi)
            logit_pi[slots < self._n_anchor_] = 50.0
            k, j = np.nonzero(U.T)
            ptr = np.zeros(U.shape[1] + 1, np.int64)
            np.cumsum(np.bincount(k, minlength=U.shape[1]), out=ptr[1:])
            with np.errstate(over="ignore"):
                s = seeds ^ np.uint64((di + 1) * 0x9E3779B97F4A7C15 % 2**64)
            if d.get("lam") is not None:          # per-component rates (noisy_or)
                s_row = np.log1p(-np.clip(d["lam"], 0.0, 1.0 - 1e-12))
                z = project_activations_ls(V, U, s_row, float(np.log1p(-d["b"])), logit_pi, ptr,
                                           j.astype(np.int64), s, 40, 20)
            else:
                T1, T0 = loglik_tables(self.likelihood, d["a"], d["b"], U.shape[1])
                z = project_activations(V, U, T1, T0, logit_pi, ptr, j.astype(np.int64), s, 40,
                                        20)
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
        if self.likelihood == "noisy_or":
            weight = getattr(self, "detection_rate_per_component_", np.full(M.shape[0], a))
            weight = np.where(np.isnan(weight), a, weight)
        else:
            weight = np.ones(M.shape[0])
        for k in range(M.shape[0]):
            q = np.clip(np.outer(Zp[:, k], M[k]) * weight[k], 0.0, 1.0 - 1e-12)
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
        """Output names for ``transform``: ``bayesianbooleanmf0``, ``bayesianbooleanmf1``, ..."""
        check_is_fitted(self, "components_")
        return np.asarray([f"bayesianbooleanmf{i}" for i in range(self.components_.shape[0])],
                          dtype=object)
