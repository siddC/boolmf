"""Likelihood engines: the parts of a sweep that depend on how rates enter the likelihood.

``CountEngine``: rates shared by all components (global or per sample). The state is the count
matrix C and the likelihood enters through tables T[row, count]; serves both likelihoods.
With ``likelihood_power`` zeta < 1 (a coarsened posterior) every update targets
likelihood^zeta x prior: the tables and the count histograms behind the rate updates are
multiplied by zeta. The log-likelihood it reports, and the predictive probabilities, are not.

``LogSurvEngine``: ``noisy_or`` with per-component detection rates (optionally also per-sample
detection and background rates). The state is L[i, j] = sum over covering components of
log(1 - lambda_ik); see ``logsurv``.

Both expose the same methods, so ``chain.run_chain`` does not depend on the rate model.
"""

import numpy as np

from .._model import (
    histogram_loglik,
    loglik_tables,
    slice_sample_unit,
    truncated_beta,
    update_rates,
)
from .kernels import (
    accumulate_entries,
    count_histograms,
    counts_from_state,
    row_histograms,
    update_activations,
    update_memberships,
)
from .logsurv import (
    accumulate_entries_ls,
    log1m_sigmoid,
    logsurv_from_state,
    survival_matrix,
    total_loglik_ls,
    update_activations_ls,
    update_component_rates,
    update_memberships_ls,
    update_sample_backgrounds,
    update_sample_offsets,
)
from .rates import SampleRates, likelihood_code, rows_loglik, slice_sample_real, table_rows
from .splitmerge import split_merge_moves, split_merge_moves_logsurv

SPREAD_SCALE = 1.0          # half-Cauchy scale of every logit-scale spread


def _seed(rng):
    return np.uint64(rng.integers(0, 2**63 - 1))


def _csr(B):
    K = B.shape[1]
    k, r = np.nonzero(B.T)
    ptr = np.zeros(K + 1, np.int64)
    np.cumsum(np.bincount(k, minlength=K), out=ptr[1:])
    return ptr, r.astype(np.int64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _update_spread(values, center, sigma, rng):
    """Slice-sample the spread of Normal(center, spread^2) values; half-Cauchy(0, 1) prior."""
    ss = float(np.sum((values - center) ** 2))
    n = values.size

    def log_logsigma(t):
        s = np.exp(t)
        return -n * t - 0.5 * ss / s**2 - np.log1p((s / SPREAD_SCALE) ** 2) + t

    return float(np.exp(slice_sample_real(log_logsigma, np.log(sigma), rng)))


def _update_center(values, sigma, prior, current, rng):
    """Slice-sample the center mu of Normal(mu, sigma^2) values; ``prior`` is on sigmoid(mu)."""
    if values.size == 0:
        return current

    def log_mu(x):
        m = np.log(x) - np.log1p(-x)
        return prior.logpdf(x) - 0.5 * np.sum((values - m) ** 2) / sigma**2

    p = slice_sample_unit(log_mu, float(_sigmoid(current)), rng)
    return float(np.log(p) - np.log1p(-p))


def _tied_rate(H1, H0, prior_a, estimation, rng):
    """``or_flip`` with background = 1 - detection (the OrMachine's symmetric flip noise).

    An entry is reconstructed correctly when it is present and covered or absent and not
    covered; the detection rate is the probability of that. ``mle`` returns the fraction
    correct (Rukat et al. 2017, Eq. 5); ``bayes`` draws from Beta(prior + counts). Both stay
    at or above 1/2, where the flip noise keeps its meaning.
    """
    if prior_a.fixed is not None:
        return prior_a.fixed
    correct = H1[1:].sum() + H0[0]
    wrong = H0[1:].sum() + H1[0]
    if estimation == "mle":
        return float(np.clip(correct / max(correct + wrong, 1), 0.5, 1.0 - 1e-12))
    aa, ab = prior_a.beta_ab or (1.0, 1.0)
    return truncated_beta(rng, aa + correct, ab + wrong, lo=0.5)


def _mle_rates(H1, H0, prior_a, prior_b):
    """``or_flip`` maximum-likelihood rates: the present fraction among covered entries and
    among uncovered entries (fixed rates stay fixed)."""
    n11, n01, n10, n00 = H1[1:].sum(), H0[1:].sum(), H1[0], H0[0]
    a = prior_a.fixed if prior_a.fixed is not None else n11 / max(n11 + n01, 1)
    b = prior_b.fixed if prior_b.fixed is not None else n10 / max(n10 + n00, 1)
    return float(np.clip(a, 1e-12, 1 - 1e-12)), float(np.clip(b, 1e-12, 1 - 1e-12))


class CountEngine:
    """Rates shared by all components (global, or per sample through ``SampleRates``)."""

    def __init__(self, cfg, V, Z, U, rng):
        n = V.shape[0]
        self.cfg, self.n, self.K = cfg, n, Z.shape[1]
        self.lik = likelihood_code(cfg.likelihood)
        self.a = cfg.prior_a.fixed if cfg.prior_a.fixed is not None else 0.9
        self.b = cfg.prior_b.fixed if cfg.prior_b.fixed is not None else 0.05
        if cfg.tied_rates:
            self.b = 1.0 - self.a
        self.det_s = SampleRates(n, self.a, cfg.prior_a) \
            if "sample" in cfg.detection_effects else None
        self.bg_s = SampleRates(n, self.b, cfg.prior_b) \
            if "sample" in cfg.background_effects else None
        self.C = counts_from_state(Z, U)
        self.power = float(cfg.likelihood_power)

    @property
    def per_sample(self):
        return self.det_s is not None or self.bg_s is not None

    def _a_vec(self):
        return self.det_s.rates if self.det_s is not None else np.full(self.n, self.a)

    def _b_vec(self):
        return self.bg_s.rates if self.bg_s is not None else np.full(self.n, self.b)

    def tables(self):
        if not self.per_sample:             # one row shared by every sample
            T1, T0 = loglik_tables(self.cfg.likelihood, self.a, self.b, self.K)
            return T1[None, :], T0[None, :]
        return table_rows(self.cfg.likelihood, self._a_vec(), self._b_vec(), self.K)

    def begin_sweep(self, Z, U):
        self.T1, self.T0 = self.tables()
        if self.power != 1.0:
            self.T1, self.T0 = self.power * self.T1, self.power * self.T0

    def _tempered(self, H1, H0):
        if self.power == 1.0:
            return H1, H0
        return self.power * H1.astype(np.float64), self.power * H0.astype(np.float64)

    def update_memberships(self, V, U, logit_rho, act_ptr, act_idx, mask, rng,
                           metropolis=False):
        update_memberships(V, U, self.C, self.T1, self.T0, logit_rho, act_ptr, act_idx, mask,
                           _seed(rng), metropolis)

    def update_activations(self, V, Z, logit_pi, mem_ptr, mem_idx, mask, rng, metropolis=False):
        update_activations(V, Z, self.C, self.T1, self.T0, logit_pi, mem_ptr, mem_idx, mask,
                           _seed(rng), metropolis)

    def split_merge(self, V, Z, U, free, zprior, uprior, n_attempts, n_launch, rng, stats):
        split_merge_moves(V, Z, U, self.C, self.T1, self.T0, free, zprior, uprior, n_attempts,
                          n_launch, rng, stats)

    def update_rates(self, V, Z, U, rng):
        """Update the rates; returns (population detection, population background, loglik)."""
        cfg = self.cfg
        if not self.per_sample:
            H1, H0 = count_histograms(V, self.C, self.K)
            T1, T0 = self._tempered(H1, H0)
            if cfg.tied_rates:
                self.a = _tied_rate(T1, T0, cfg.prior_a, cfg.rate_estimation, rng)
                self.b = 1.0 - self.a
            elif cfg.rate_estimation == "mle":
                self.a, self.b = _mle_rates(H1, H0, cfg.prior_a, cfg.prior_b)
            else:
                self.a, self.b = update_rates(cfg.likelihood, self.a, self.b, T1, T0,
                                              cfg.prior_a, cfg.prior_b, rng)
            return self.a, self.b, histogram_loglik(cfg.likelihood, self.a, self.b, H1, H0)
        n, lik = self.n, self.lik
        H1, H0 = row_histograms(V, self.C, self.K)
        T1, T0 = self._tempered(H1, H0)
        order = cfg.likelihood == "or_flip"
        zeros, ones = np.zeros(n), np.ones(n)
        bv = self._b_vec()
        if self.det_s is not None:
            self.det_s.update(lik, 0, bv, T1, T0, bv if order else zeros, ones, rng)
            self.a = self.det_s.population_rate
        elif cfg.prior_a.fixed is None:
            lo = float(bv.max()) if order else 0.0
            if order and cfg.prior_a.beta_ab:
                aa, ab = cfg.prior_a.beta_ab
                self.a = truncated_beta(rng, aa + T1[:, 1:].sum(), ab + T0[:, 1:].sum(), lo=lo)
            else:
                def la(x):
                    if x <= lo:
                        return -np.inf
                    return rows_loglik(lik, np.full(n, x), bv, T1, T0) + cfg.prior_a.logpdf(x)

                self.a = slice_sample_unit(la, max(self.a, lo + 1e-9), rng)
        av = self._a_vec()
        if self.bg_s is not None:
            self.bg_s.update(lik, 1, av, T1, T0, zeros, av if order else ones, rng)
            self.b = self.bg_s.population_rate
        elif cfg.prior_b.fixed is None:
            hi = float(av.min()) if order else 1.0
            if order and cfg.prior_b.beta_ab:
                ba, bb = cfg.prior_b.beta_ab
                self.b = truncated_beta(rng, ba + T1[:, 0].sum(), bb + T0[:, 0].sum(), hi=hi)
            else:
                def lb(x):
                    if x >= hi:
                        return -np.inf
                    return rows_loglik(lik, av, np.full(n, x), T1, T0) + cfg.prior_b.logpdf(x)

                self.b = slice_sample_unit(lb, min(self.b, hi - 1e-9), rng)
        return self.a, self.b, float(rows_loglik(lik, av, self._b_vec(), H1, H0))

    def accumulate(self, Z, U, explained, predictive):
        T1, _ = self.tables()
        accumulate_entries(self.C, T1, explained, predictive)

    def draw_extras(self):
        """Per-sample rates and spreads to keep with a draw (None when rates are global)."""
        out = {}
        if self.det_s is not None:
            out["detection"] = self.det_s.rates.astype(np.float32)
        if self.bg_s is not None:
            out["background"] = self.bg_s.rates.astype(np.float32)
        spread = (self.det_s.sigma if self.det_s is not None else np.nan,
                  self.bg_s.sigma if self.bg_s is not None else np.nan, np.nan)
        return out, (spread if self.per_sample else None)

    def slot_rates(self, slots):
        return None


class LogSurvEngine:
    """``noisy_or`` with a detection rate per component (and optionally per sample).

    logit lambda_ik = off_i + h_k. Without per-sample detection rates off_i = 0 and
    h_k ~ Normal(mu, tau^2), sigmoid(mu) being the population rate (prior ``detection_prior``).
    With them off_i = y_i ~ Normal(mu, sigma^2) and h_k ~ Normal(0, tau^2). tau and sigma have
    half-Cauchy(0, 1) priors. The background rate is global or per sample (logit-normal).
    """

    def __init__(self, cfg, V, Z, U, rng):
        n, F = V.shape
        self.cfg, self.n, self.F, self.K = cfg, n, F, Z.shape[1]
        a0 = 0.9
        self.sample_det = "sample" in cfg.detection_effects
        self.mu = float(np.log(a0) - np.log1p(-a0))
        self.sigma = 0.5
        self.tau = 0.5
        if self.sample_det:
            self.off = np.full(n, self.mu)
            self.h = np.zeros(self.K)
        else:
            self.off = np.zeros(1)
            self.h = np.full(self.K, self.mu)
        b0 = cfg.prior_b.fixed if cfg.prior_b.fixed is not None else 0.05
        self.sample_bg = "sample" in cfg.background_effects
        self.b = b0
        self.mu_b = float(np.log(b0) - np.log1p(-b0))
        self.sigma_b = 0.5
        self.yb = np.full(n, self.mu_b) if self.sample_bg else None
        self._set_lb0()
        self.S = survival_matrix(self.off, self.h)
        self.L = None

    def _set_lb0(self):
        if self.sample_bg:
            self.LB0 = np.array([log1m_sigmoid(y) for y in self.yb])
        else:
            self.LB0 = np.array([np.log1p(-self.b)])

    @property
    def a(self):
        return float(_sigmoid(self.mu))

    def begin_sweep(self, Z, U):
        self.S = survival_matrix(self.off, self.h)
        mem_ptr, mem_idx = _csr(U)
        self.L = logsurv_from_state(Z, self.S, mem_ptr, mem_idx, self.F)

    def update_memberships(self, V, U, logit_rho, act_ptr, act_idx, mask, rng,
                           metropolis=False):
        update_memberships_ls(V, U, self.L, self.LB0, self.S, logit_rho, act_ptr, act_idx, mask,
                              _seed(rng), metropolis)

    def update_activations(self, V, Z, logit_pi, mem_ptr, mem_idx, mask, rng, metropolis=False):
        update_activations_ls(V, Z, self.L, self.LB0, self.S, logit_pi, mem_ptr, mem_idx, mask,
                              _seed(rng), metropolis)

    def split_merge(self, V, Z, U, free, zprior, uprior, n_attempts, n_launch, rng, stats):
        split_merge_moves_logsurv(V, Z, U, self.L, self.S, self.LB0, free, zprior, uprior,
                                  n_attempts, n_launch, rng, stats)

    def update_rates(self, V, Z, U, rng):
        cfg = self.cfg
        act_ptr, act_idx = _csr(Z)
        mem_ptr, mem_idx = _csr(U)
        used = (np.diff(act_ptr) > 0) & (np.diff(mem_ptr) > 0)
        mean_h = 0.0 if self.sample_det else self.mu
        update_component_rates(V, Z, U, self.L, self.LB0, self.S, self.off, self.h, mean_h,
                               self.tau, act_ptr, act_idx, mem_ptr, mem_idx, used, _seed(rng))
        hu = self.h[used]
        if hu.size:
            self.tau = _update_spread(hu, mean_h, self.tau, rng)
        if self.sample_det:
            update_sample_offsets(V, Z, self.L, self.LB0, self.S, self.off, self.h, self.mu,
                                  self.sigma, mem_ptr, mem_idx, _seed(rng))
            self._shift_offsets(used, rng)
            self.mu = _update_center(self.off, self.sigma, cfg.prior_a, self.mu, rng)
            self.sigma = _update_spread(self.off, self.mu, self.sigma, rng)
        else:
            self.mu = _update_center(hu, self.tau, cfg.prior_a, self.mu, rng)
        if self.sample_bg:
            update_sample_backgrounds(V, self.L, self.yb, self.mu_b, self.sigma_b, _seed(rng))
            self._set_lb0()
            self.mu_b = _update_center(self.yb, self.sigma_b, cfg.prior_b, self.mu_b, rng)
            self.sigma_b = _update_spread(self.yb, self.mu_b, self.sigma_b, rng)
            self.b = float(_sigmoid(self.mu_b))
        elif cfg.prior_b.fixed is None:
            L, prior = self.L, cfg.prior_b

            def lb(x):
                return total_loglik_ls(V, L, np.array([np.log1p(-x)])) + prior.logpdf(x)

            self.b = slice_sample_unit(lb, self.b, rng)
            self._set_lb0()
        return self.a, self.b, float(total_loglik_ls(V, self.L, self.LB0))

    def _shift_offsets(self, used, rng):
        """Exact move along the direction the likelihood cannot see: mu + c, y_i + c, h_k - c
        leaves every lambda_ik and every y_i - mu unchanged, so c is drawn from what is left of
        the prior (the component offsets and the population rate) by slice sampling."""
        hu = self.h[used]
        tau, mu, prior = self.tau, self.mu, self.cfg.prior_a

        def logdens(c):
            x = 1.0 / (1.0 + np.exp(-(mu + c)))
            if not 0.0 < x < 1.0:
                return -np.inf
            return (-0.5 * np.sum((hu - c) ** 2) / tau**2 + prior.logpdf(x)
                    + np.log(x) + np.log1p(-x))

        c = slice_sample_real(logdens, 0.0, rng)
        self.mu += c
        self.off += c
        self.h[used] -= c

    def accumulate(self, Z, U, explained, predictive):
        accumulate_entries_ls(counts_from_state(Z, U), self.L, self.LB0, explained, predictive)

    def draw_extras(self):
        out = {}
        if self.sample_det:
            out["detection"] = _sigmoid(self.off).astype(np.float32)
        if self.sample_bg:
            out["background"] = _sigmoid(self.yb).astype(np.float32)
        spread = (self.sigma if self.sample_det else np.nan,
                  self.sigma_b if self.sample_bg else np.nan, self.tau)
        return out, spread

    def slot_rates(self, slots):
        """Population-level detection rate of each slot: sigmoid(mu + h_k) or sigmoid(h_k)."""
        center = self.mu if self.sample_det else 0.0
        return _sigmoid(center + self.h[slots])

    def population_survival(self, slots):
        return np.log1p(-self.slot_rates(slots))


def make_engine(cfg, V, Z, U, rng):
    if "component" in cfg.detection_effects:
        return LogSurvEngine(cfg, V, Z, U, rng)
    return CountEngine(cfg, V, Z, U, rng)
