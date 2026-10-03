"""One MCMC chain for BayesianBooleanMF: initialization, burn-in, sampling and draw storage."""

import os
import pickle
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.special import betaln, expit

from .._model import loglik_tables
from ..diagnostics import ess, geweke, segment_rhat
from .collapsed import BIRTH_ENUMERATE, BIRTH_METROPOLIS, collapsed_rows
from .engines import make_engine
from .splitmerge import N_MOVES

MONITORED = ("log_likelihood", "n_active", "detection_rate", "background_rate", "alpha")
# alpha mixes slowly and is reported, but it does not gate the end of burn-in
BURN_MONITORED = ("n_active", "detection_rate", "background_rate")


@dataclass
class ChainConfig:
    likelihood: str
    n_slots: int                      # anchors + free slots
    anchor_members: list              # list of bool arrays (n_features,)
    anchor_learned: list              # list of bool, membership learned?
    nonparametric: bool
    prior_a: object                   # RatePrior for the detection rate
    prior_b: object                   # RatePrior for the background rate
    membership_ab: tuple              # Beta(a, b) prior on rho_k
    alpha_prior: object               # PositivePrior on alpha
    burn_in: object                   # "auto" or int
    max_sweeps: int
    n_draws: int
    thin: object                      # "auto" or int
    rhat_threshold: float
    n_init: int
    store_draws: object = True         # True, False, or the most draws to keep (evenly spaced)
    track_map: bool = False           # keep the draw with the highest log posterior
    checkpoint_dir: str = None        # save the chain's state here and resume from it
    checkpoint_every: int = 200       # sweeps between checkpoints
    data_key: str = ""                # fingerprint of the data, to refuse foreign checkpoints
    store_entries: bool = True
    check_every: int = 100
    min_burn: int = 500
    n_threads: int = 0
    min_support: int = 3
    burn_rhat: float = 1.05
    n_split_merge: int = 10           # split-merge attempts per sweep (0 = off)
    n_launch: int = 4                 # restricted Gibbs scans that build each launch state
    activation_prior: object = None   # LevelPrior for Z (None: IBP or Beta(1, 1) per component)
    membership_prior: object = None   # LevelPrior for U (None: Beta(membership_ab) per component)
    tied_rates: bool = False          # or_flip with background = 1 - detection
    rate_estimation: str = "bayes"    # "bayes" (sample) or "mle" (point estimate every sweep)
    metropolis: bool = False          # Metropolised Gibbs flips (Liu 1996) instead of Gibbs draws
    activations_first: bool = False   # update Z before U in each sweep
    init_kind: str = "random"         # "random", "uniform" or "empty" when no init tuple is given
    births: str = "slots"             # "slots", "enumerate" (Wood 2006), "metropolis" (Meeds)
    max_new: int = 10                 # most new components one row may add at once
    exact_birth_members: bool = True  # False: one Gibbs pass from 0 over new memberships
    detection_effects: tuple = ()     # subset of ("sample", "component")
    background_effects: tuple = ()    # ("sample",) for per-sample background rates
    verbose: int = 0


@dataclass
class ChainResult:
    seed: int
    trace: dict
    burn_in: int
    thin: int
    converged: bool
    member_mean: np.ndarray           # (n_features, n_slots) float32
    activation_mean: np.ndarray       # (n_samples, n_slots) float32
    explained: np.ndarray = None      # (n_samples, n_features) float32
    predictive: np.ndarray = None     # (n_samples, n_features) float32
    draws: list = field(default_factory=list)
    draw_rates: np.ndarray = None     # (n_draws, 2)
    draw_alpha: np.ndarray = None
    draw_n_active: np.ndarray = None
    draw_loglik: np.ndarray = None
    split_merge: np.ndarray = None    # (2, N_MOVES): attempts and acceptances per move type
    births: np.ndarray = None         # collapsed sampler: births, deaths, truncation hits
    draw_sample_rates: dict = None    # "detection" / "background": (n_draws, n_samples) float32
    draw_slot_rates: list = None      # per draw (slots, detection rate per slot) or None
    draw_spread: np.ndarray = None    # (n_draws, 3) logit-scale spreads: sample detection,
    #                                   sample background, component detection
    map_state: object = None          # (log posterior, Z, U, a, b) with track_map


def _csr(B):
    """Column-wise index lists of a binary matrix B (rows x K) -> (ptr, idx)."""
    K = B.shape[1]
    k, r = np.nonzero(B.T)
    ptr = np.zeros(K + 1, np.int64)
    np.cumsum(np.bincount(k, minlength=K), out=ptr[1:])
    return ptr, r.astype(np.int64)


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def initial_state(V, cfg, rng, init):
    n, F = V.shape
    K = cfg.n_slots
    na = len(cfg.anchor_members)
    Z = np.zeros((n, K), np.int8)
    U = np.zeros((F, K), np.int8)
    for a, members in enumerate(cfg.anchor_members):
        Z[:, a] = 1
        U[:, a] = members
    if isinstance(init, tuple):
        members, activations = init
        k0 = min(members.shape[0], K - na)
        U[:, na:na + k0] = members[:k0].T.astype(np.int8)
        Z[:, na:na + k0] = activations[:, :k0].astype(np.int8)
    elif cfg.init_kind == "uniform":          # every free entry 0 or 1 with probability 1/2
        U[:, na:] = rng.random((F, K - na)) < 0.5
        Z[:, na:] = rng.random((n, K - na)) < 0.5
    elif cfg.init_kind == "random":
        for k in range(na, min(K, na + cfg.n_init)):
            U[:, k] = rng.random(F) < 0.1
            Z[:, k] = rng.random(n) < 0.3
    return Z, U                                # "empty": no free component to start with


@dataclass
class LevelPrior:
    """Prior on activation (or membership) probabilities.

    kind "ibp" (activations only), "beta" (rate ~ Beta(a, b)), "fixed" (rate = value) or
    "mixture" (rate ~ Beta(b1, b2) if psi = 1 else Beta(c1, c2), psi ~ Bernoulli(w),
    w ~ Beta(d1, d2), with ``mix`` = (b1, b2, c1, c2, d1, d2)); level "component" (one rate per
    component), "shared" (one rate for all components) or "row" (one rate per sample, or per
    feature).
    """

    kind: str = "beta"
    level: str = "component"
    a: float = 1.0
    b: float = 1.0
    value: float = 0.5
    mix: tuple = None


def _units(prior, X, cols, n_rows):
    """(ones, trials) per rate unit of a binary matrix X (rows x slots) over ``cols``."""
    Xc = X[:, cols]
    m = len(cols)
    if prior.level == "component":
        return Xc.sum(0, dtype=np.int64), np.full(m, n_rows)
    if prior.level == "shared":
        return np.array([int(Xc.sum())]), np.array([n_rows * m])
    return Xc.sum(1, dtype=np.int64), np.full(Xc.shape[0], m)


def _mixture_rates(prior, ones, trials, state, rng):
    """One Gibbs pass over a two-component Beta mixture prior: the rates given the current
    indicators psi, then the weight w given psi, then psi given the rates and w (the order of
    Wagala et al. 2026, Algorithm 1). ``state`` holds psi and w between sweeps."""
    b1, b2, c1, c2, d1, d2 = prior.mix
    if "psi" not in state:
        state["psi"] = rng.random(ones.shape[0]) < 0.5
    psi = state["psi"]
    rates = np.where(psi, rng.beta(b1 + ones, b2 + trials - ones),
                     rng.beta(c1 + ones, c2 + trials - ones))
    rates = np.clip(rates, 1e-300, 1 - 1e-16)
    w = rng.beta(d1 + psi.sum(), d2 + psi.size - psi.sum())
    w = min(max(w, 1e-300), 1 - 1e-16)
    l1 = _log_beta_pdf(rates, b1, b2) + np.log(w)
    l0 = _log_beta_pdf(rates, c1, c2) + np.log1p(-w)
    state["psi"] = rng.random(rates.size) < expit(l1 - l0)
    state["w"] = w
    return rates


def _log_posterior(cfg, ll, Z, U, pi, rho, apri, mpri, mix_z, mix_u, a, b, free):
    """Unnormalized log posterior of a finite model with global rates (the constant of a rate
    prior given as a Beta distribution is left out; it does not change which draw is best)."""
    lp = ll
    for prior, x in ((cfg.prior_a, a), (cfg.prior_b, b)):
        if prior.fixed is None:
            lp += float(prior.logpdf(x))
    zprior = apri or LevelPrior("beta", "component", 1.0, 1.0)
    for X, p, prior, state in ((Z[:, free], pi[:, free], zprior, mix_z),
                               (U[:, free], rho[:, free], mpri, mix_u)):
        p = np.clip(p, 1e-300, 1 - 1e-16)
        lp += float((X * np.log(p) + (1 - X) * np.log1p(-p)).sum())
        if prior.kind in ("fixed", "ibp"):
            continue
        rates = p[0, :] if prior.level == "component" else (
            p[:, 0] if prior.level == "row" else p[:1, 0])
        if prior.kind == "beta":
            lp += float(_log_beta_pdf(rates, prior.a, prior.b).sum())
        else:
            b1, b2, c1, c2, d1, d2 = prior.mix
            psi, w = state["psi"], state["w"]
            lp += float(np.where(psi, _log_beta_pdf(rates, b1, b2),
                                 _log_beta_pdf(rates, c1, c2)).sum())
            lp += float(psi.sum() * np.log(w) + (psi.size - psi.sum()) * np.log1p(-w))
            lp += float(_log_beta_pdf(np.array([w]), d1, d2)[0])
    return lp


def _log_beta_pdf(x, a, b):
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b)


def _level_rates(prior, X, cols, rng, n_rows, state=None):
    """Draw the probabilities of a binary matrix X (rows x slots) restricted to ``cols``.

    Returns an array broadcastable to (rows, len(cols)): (1, m) or (rows, 1) or (1, 1).
    """
    m = len(cols)
    if prior.kind == "fixed":
        return np.full((1, 1), prior.value)
    if prior.kind == "mixture":
        ones, trials = _units(prior, X, cols, n_rows)
        rates = _mixture_rates(prior, ones, trials, state, rng)
        if prior.level == "component":
            return rates[None, :]
        return rates[:, None] if prior.level == "row" else rates.reshape(1, 1)
    Xc = X[:, cols]
    if prior.level == "component":
        ones = Xc.sum(0, dtype=np.int64)
        return rng.beta(prior.a + ones, prior.b + n_rows - ones)[None, :]
    if prior.level == "shared":
        ones = int(Xc.sum())
        return np.full((1, 1), rng.beta(prior.a + ones, prior.b + n_rows * m - ones))
    ones = Xc.sum(1, dtype=np.int64)                     # "row"
    return rng.beta(prior.a + ones, prior.b + m - ones)[:, None]


_STATE = ("rng", "Z", "U", "engine", "rho", "pi", "alpha", "mix_z", "mix_u", "best_map", "trace",
          "burn", "converged", "thin", "Ubar", "Zbar", "explained", "predictive", "draws",
          "d_rates", "d_alpha", "d_nact", "d_ll", "d_srates", "d_spread", "d_slot_rates", "n_kept",
          "sm_stats", "birth_stats", "sweep", "sampling_start", "passes")
_FORMAT = 1


def _checkpoint_path(cfg, seed):
    if not cfg.checkpoint_dir:
        return None
    return os.path.join(cfg.checkpoint_dir, f"chain-{int(seed)}.pkl")


def _canon(v):
    """A description of a setting that is stable across processes (no memory addresses)."""
    if isinstance(v, np.ndarray):
        return ("array", v.dtype.str, v.shape, v.tobytes().hex())
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return tuple((k, _canon(x)) for k, x in sorted(v.items()))
    if hasattr(v, "dist") and hasattr(v, "args") and hasattr(v, "kwds"):   # frozen scipy
        return ("dist", v.dist.name, _canon(v.args), _canon(v.kwds))
    if hasattr(v, "__dict__"):
        return (type(v).__name__, _canon(vars(v)))
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def _config_key(cfg):
    """Everything that defines the chain except where and how often it is checkpointed."""
    skip = {"checkpoint_dir", "checkpoint_every", "n_threads", "verbose"}
    return repr(_canon({k: v for k, v in vars(cfg).items() if k not in skip}))


def _save_checkpoint(path, cfg, seed, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"format": _FORMAT, "seed": int(seed), "config": _config_key(cfg),
                     "state": state}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)                      # atomic: a crash never leaves half a file


def _load_checkpoint(path, cfg, seed):
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    if saved.get("format") != _FORMAT or saved.get("seed") != int(seed) \
            or saved.get("config") != _config_key(cfg):
        raise ValueError(
            f"checkpoint {path} was written for different data or settings; remove it or use "
            "another checkpoint_dir.")
    return saved["state"]


def run_chain(V, cfg, seed, init):
    """Run one chain to completion and return its summaries and draws."""
    if cfg.n_threads:
        numba.set_num_threads(max(1, min(cfg.n_threads, numba.config.NUMBA_NUM_THREADS)))
    rng = np.random.default_rng(seed)
    n, F = V.shape
    K = cfg.n_slots
    na = len(cfg.anchor_members)
    free = np.zeros(K, bool)
    free[na:] = True
    Kf = max(1, K - na)
    mem_mask = free.copy()
    for a, learned in enumerate(cfg.anchor_learned):
        mem_mask[a] = bool(learned)
    act_mask = free.copy()

    ra, rb = cfg.membership_ab
    Z, U = initial_state(V, cfg, rng, init)
    engine = make_engine(cfg, V, Z, U, rng)
    rho = np.clip(U.mean(0), 1e-4, 1 - 1e-4).astype(np.float64)[None, :]
    pi = np.clip(Z.mean(0), 1e-6, 1 - 1e-6).astype(np.float64)[None, :]
    pi[:, ~free] = 1.0
    apri = cfg.activation_prior
    mpri = cfg.membership_prior or LevelPrior("beta", "component", ra, rb)
    if apri is not None and apri.kind == "fixed":
        pi[:, free] = apri.value
    if mpri.kind == "fixed":
        rho = np.full((1, K), mpri.value)
    elif cfg.births != "slots":
        rho = np.full((1, K), mpri.a / (mpri.a + mpri.b))
    alpha = cfg.alpha_prior.fixed if cfg.alpha_prior.fixed is not None else 1.0
    mix_z, mix_u = {}, {}                      # mixture-prior indicators and weights
    best_map = None                            # (log posterior, Z, U, a, b) of the best draw
    keep_every = 1
    if cfg.store_draws is not True and cfg.store_draws and isinstance(cfg.n_draws, int):
        keep_every = max(1, -(-cfg.n_draws // int(cfg.store_draws)))

    total_cap = cfg.max_sweeps
    trace = {m: [] for m in MONITORED}
    burn = cfg.burn_in if isinstance(cfg.burn_in, int) else None
    converged = isinstance(cfg.burn_in, int)
    thin = cfg.thin if isinstance(cfg.thin, int) else None

    Ubar = np.zeros((F, K), np.float32)
    Zbar = np.zeros((n, K), np.float32)
    explained = np.zeros((n, F), np.uint16) if cfg.store_entries else None
    predictive = np.zeros((n, F), np.float32) if cfg.store_entries else None
    draws, d_rates, d_alpha, d_nact, d_ll = [], [], [], [], []
    d_srates = {"detection": [], "background": []}
    d_spread = []
    d_slot_rates = []
    n_kept = 0
    sm_stats = np.zeros((2, N_MOVES), np.int64)
    birth_stats = np.zeros(3, np.int64)
    harmonic = float(np.sum(1.0 / np.arange(1, n + 1)))
    sweep = 0
    sampling_start = None
    passes = 0

    ckpt = _checkpoint_path(cfg, seed)
    if ckpt is not None and os.path.exists(ckpt):
        state = _load_checkpoint(ckpt, cfg, seed)
        (rng, Z, U, engine, rho, pi, alpha, mix_z, mix_u, best_map, trace, burn, converged, thin,
         Ubar, Zbar, explained, predictive, draws, d_rates, d_alpha, d_nact, d_ll, d_srates,
         d_spread, d_slot_rates, n_kept, sm_stats, birth_stats, sweep, sampling_start,
         passes) = (state[k] for k in _STATE)
    resumed_at = sweep

    while True:
        if ckpt is not None and sweep != resumed_at and sweep % cfg.checkpoint_every == 0:
            local = locals()
            _save_checkpoint(ckpt, cfg, seed, {k: local[k] for k in _STATE})
        # ---- one Gibbs sweep ---------------------------------------------------------
        engine.begin_sweep(Z, U)
        if cfg.births != "slots":
            alpha, rho, pi = _collapsed_sweep(cfg, V, Z, U, engine, alpha, rho, mpri, harmonic,
                                              birth_stats, rng)
            nmem = U.sum(0, dtype=np.int64)
            nact = Z.sum(0, dtype=np.int64)
        elif cfg.activations_first:
            mem_ptr, mem_idx = _csr(U)
            engine.update_activations(V, Z, _logit(pi), mem_ptr, mem_idx, act_mask, rng,
                                      cfg.metropolis)
            act_ptr, act_idx = _csr(Z)
            engine.update_memberships(V, U, _logit(rho), act_ptr, act_idx, mem_mask, rng,
                                      cfg.metropolis)
        else:
            act_ptr, act_idx = _csr(Z)
            engine.update_memberships(V, U, _logit(rho), act_ptr, act_idx, mem_mask, rng,
                                      cfg.metropolis)
            nmem = U.sum(0, dtype=np.int64)
            phantom = free & (nmem == 0)
            if phantom.any() and cfg.nonparametric:
                Z[:, phantom] = 0                  # memberless slots carry no carriers
            mem_ptr, mem_idx = _csr(U)
            engine.update_activations(V, Z, _logit(pi), mem_ptr, mem_idx, act_mask, rng,
                                      cfg.metropolis)
        nact = Z.sum(0, dtype=np.int64)
        orphan = free & (nact == 0)
        if orphan.any() and cfg.nonparametric:
            U[:, orphan] = 0                       # carrier-less slots keep no members
        nmem = U.sum(0, dtype=np.int64)
        if cfg.births == "slots" and cfg.n_split_merge > 0:
            phantom = free & (nmem == 0)
            if phantom.any():
                Z[:, phantom] = 0                  # so every free slot is used or empty
            zprior = (alpha / Kf, 1.0) if cfg.nonparametric else \
                ((apri.a, apri.b) if apri is not None else (1.0, 1.0))
            engine.split_merge(V, Z, U, free, zprior, (ra, rb), cfg.n_split_merge,
                               cfg.n_launch, rng, sm_stats)
            nmem = U.sum(0, dtype=np.int64)
            nact = Z.sum(0, dtype=np.int64)

        if cfg.births != "slots":
            pass                                  # updated inside the collapsed sweep
        elif mpri.kind != "fixed":
            if mpri.level == "component" and mpri.kind == "beta":
                rho = rng.beta(ra + nmem, rb + F - nmem).astype(np.float64)[None, :]
            else:                               # shared or per feature, over the used slots
                cols = np.flatnonzero((nmem > 0) & (nact > 0) | ~free) \
                    if cfg.nonparametric else np.arange(K)
                rho = np.broadcast_to(_level_rates(mpri, U, cols, rng, F, mix_u),
                                      (F if mpri.level == "row" else 1, K)).copy()
        if cfg.births != "slots":
            pass
        elif cfg.nonparametric:
            pi_f = rng.beta(alpha / Kf + nact[free], 1.0 + n - nact[free])
            pi_f = np.clip(pi_f, 1e-300, 1 - 1e-12)
            if cfg.alpha_prior.fixed is None:
                rate = cfg.alpha_prior.rate - np.log(pi_f).sum() / Kf
                alpha = rng.gamma(cfg.alpha_prior.shape + Kf, 1.0 / rate)
            pi = np.ones((1, K))
            pi[0, free] = pi_f
        elif apri is None or apri.kind != "fixed":
            prior_z = apri or LevelPrior("beta", "component", 1.0, 1.0)
            rates = _level_rates(prior_z, Z, np.flatnonzero(free), rng, n, mix_z)
            pi = np.ones((n if prior_z.level == "row" else 1, K))
            pi[:, free] = rates

        a, b, ll = engine.update_rates(V, Z, U, rng)
        n_active = int((free & (nmem >= cfg.min_support) & (nact >= cfg.min_support)).sum())
        trace["log_likelihood"].append(ll)
        trace["n_active"].append(n_active)
        trace["detection_rate"].append(a)
        trace["background_rate"].append(b)
        trace["alpha"].append(alpha if cfg.nonparametric else np.nan)
        sweep += 1

        # ---- burn-in control -------------------------------------------------------------
        if sampling_start is None:
            end_burn = False
            if burn is not None:
                end_burn = sweep >= burn
            elif sweep >= cfg.min_burn and sweep % cfg.check_every == 0:
                w = max(cfg.min_burn, sweep // 2)
                ok = True
                for m in BURN_MONITORED:
                    x = np.asarray(trace[m][-w:], float)
                    if np.all(np.isnan(x)):
                        continue
                    if segment_rhat(x, 4) > cfg.burn_rhat:
                        ok = False
                        break
                if ok and geweke(np.asarray(trace["log_likelihood"][-w:], float)) > 2.0:
                    ok = False                       # log-likelihood still trending
                passes = passes + 1 if ok else 0
                if passes >= 2:                      # two consecutive passing checks
                    converged = True
                    end_burn = True
            if not end_burn and sweep >= total_cap - cfg.n_draws:
                end_burn = True                      # budget exhausted, sample anyway
            if end_burn:
                burn = sweep
                if thin is None:
                    w = max(20, min(sweep // 2, 1000))
                    x = np.asarray(trace["log_likelihood"][-w:], float)
                    e = ess(x[None, :])
                    tau = w / e if np.isfinite(e) and e > 0 else 1.0
                    thin = int(np.clip(np.ceil(tau), 1, 50))
                budget = max(1, (total_cap - burn) // cfg.n_draws)
                thin = max(1, min(thin, budget))
                sampling_start = sweep
            continue

        # ---- sampling ---------------------------------------------------------------------
        if (sweep - sampling_start) % thin != 0:
            continue
        Ubar += U
        Zbar += Z
        if cfg.store_entries:
            engine.accumulate(Z, U, explained, predictive)
        extras, spread = engine.draw_extras()
        for kind, v in extras.items():
            d_srates[kind].append(v)
        if spread is not None:
            d_spread.append(spread)
        used = np.flatnonzero((nmem > 0) & (nact > 0) | ~free)
        lam = engine.slot_rates(used)
        if lam is not None:
            d_slot_rates.append((used.astype(np.int32), lam.astype(np.float64)))
        if cfg.track_map:
            lp = _log_posterior(cfg, ll, Z, U, pi, rho, apri, mpri, mix_z, mix_u, a, b, free)
            if best_map is None or lp > best_map[0]:
                best_map = (lp, Z.copy(), U.copy(), a, b)
        if cfg.store_draws and n_kept % keep_every == 0:
            draw = {
                "slots": used.astype(np.int32),
                "U": np.packbits(U[:, used], axis=0),
                "Z": np.packbits(Z[:, used], axis=0),
                "pi": pi.mean(0)[used].astype(np.float64),
                "rho": np.broadcast_to(rho, (rho.shape[0], K)).mean(0)[used].astype(np.float64),
            }
            if lam is not None:
                draw["lam"] = lam.astype(np.float64)
            draws.append(draw)
        d_rates.append((a, b))
        d_alpha.append(alpha if cfg.nonparametric else np.nan)
        d_nact.append(n_active)
        d_ll.append(ll)
        n_kept += 1
        if n_kept >= cfg.n_draws:
            break

    if ckpt is not None and os.path.exists(ckpt):
        os.remove(ckpt)                        # the chain is complete
    res = ChainResult(
        seed=int(seed),
        trace={m: np.asarray(v, float) for m, v in trace.items()},
        burn_in=int(burn),
        thin=int(thin),
        converged=bool(converged),
        member_mean=Ubar / n_kept,
        activation_mean=Zbar / n_kept,
        draws=draws,
        draw_rates=np.asarray(d_rates, float),
        draw_alpha=np.asarray(d_alpha, float),
        draw_n_active=np.asarray(d_nact, int),
        draw_loglik=np.asarray(d_ll, float),
        split_merge=sm_stats,
        births=birth_stats,
        draw_sample_rates={k: np.asarray(v, np.float32) for k, v in d_srates.items() if v},
        draw_spread=np.asarray(d_spread, float) if d_spread else None,
        draw_slot_rates=d_slot_rates or None,
        map_state=best_map,
    )
    if cfg.store_entries:
        res.explained = (explained.astype(np.float32) / n_kept)
        res.predictive = predictive / n_kept
    return res



def _collapsed_sweep(cfg, V, Z, U, engine, alpha, rho, mpri, harmonic, stats, rng):
    """One sweep of the collapsed IBP sampler (``births`` "enumerate" or "metropolis").

    Rows of Z are updated in turn with births (``collapsed.collapsed_rows``), then U by the
    parallel Gibbs kernel with the shared membership probability p, then p (when it has a Beta
    prior) and alpha (when not fixed). Returns (alpha, rho, pi) for the chain's bookkeeping.
    """
    n, F = V.shape
    K = Z.shape[1]
    p = float(rho.flat[0])
    T1, T0 = loglik_tables(cfg.likelihood, engine.a, engine.b, K + cfg.max_new + 1)
    births = BIRTH_ENUMERATE if cfg.births == "enumerate" else BIRTH_METROPOLIS
    collapsed_rows(V, Z, U, engine.C, T1, T0, float(alpha), p, births, cfg.max_new,
                   cfg.metropolis, np.uint64(rng.integers(0, 2**63 - 1)), stats,
                   cfg.exact_birth_members)
    used = Z.sum(0) > 0
    U[:, ~used] = 0
    act_ptr, act_idx = _csr(Z)
    engine.update_memberships(V, U, np.full((1, K), _logit(np.array([p]))[0]), act_ptr, act_idx,
                              used, rng, cfg.metropolis)
    k_plus = int(used.sum())
    if mpri.kind != "fixed":
        ones = int(U[:, used].sum())
        p = float(rng.beta(mpri.a + ones, mpri.b + F * k_plus - ones))
    if cfg.alpha_prior.fixed is None:
        alpha = float(rng.gamma(cfg.alpha_prior.shape + k_plus,
                                1.0 / (cfg.alpha_prior.rate + harmonic)))
    m = Z.sum(0).astype(float)
    pi = np.clip(m / n, 1e-12, 1 - 1e-12)[None, :]
    return alpha, np.full((1, K), p), pi
