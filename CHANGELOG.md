# Changelog

## 0.2.0 (unreleased)

### Added
- `benchmarks/genomicus4k.py`: the v0.2 real-data gate on the GENOMiCUS-4k E. coli pangenome
  (data not distributed; paths are passed in). Held-out entries are scored against a masked
  NMF baseline at its validation-chosen rank. The Bayesian model predicts better in every
  variant (500 genomes: log-likelihood -0.115, AUC 0.986, against -0.148 and 0.982), but the
  chains do not converge in 4,000 sweeps: they keep adding components (about 690 per draw at
  500 genomes), and the R-hat of the log-likelihood stays near 3. Per-genome rates help most
  (rate R-hat 1.18 / 1.07). With the core genome added, the inner core works as a fixed anchor
  component (held-out log-likelihood -0.093); as a learned anchor it absorbs the soft core and
  about a quarter of the accessory genes and predicts worse.
  On a phylogroup-balanced subsample (`--metadata`: 25 genomes from each of phylogroups A, B1,
  B2, C, D, E, F and G, 10 from each Shigella species and the 9 clade I genomes; 249 x 4,811)
  with per-genome rates, 4 chains of 20,000 sweeps do not settle either: the log-likelihood
  still rises by about 650 per 1,000 sweeps at the end, the component count by about 3 (to
  about 610), and the chains stay apart (log-likelihood R-hat 2.8 over the last 5,000 sweeps;
  rates 1.14 / 1.07). Only 14% of one chain's components have a partner in another chain at
  Jaccard >= 0.8 (members and carriers). The chains agree on prediction: held-out
  log-likelihood -0.119 to -0.122 per chain (-0.106 pooled) against NMF's best -0.187.
- `checkpoint_dir` and `checkpoint_every` for `BayesianBooleanMF`: each chain saves its complete
  state (sampler state, random generator, traces and accumulated draws) every
  `checkpoint_every` sweeps, written atomically, and a refit with the same data, settings and
  int `random_state` resumes from it. A resumed fit is identical to an uninterrupted one (tested
  by interrupting a chain mid-run); a checkpoint from other data or settings raises
  `ValueError`; finished chains remove their file. Long real-data fits were being lost to
  process restarts.
- `presets.wagala2026`, the BBMF of Wagala, Samur & Parmigiani (2026), and what it needs:
  - `BetaMixture`, a two-component Beta mixture prior for `membership_prior` or
    `activation_prior` (fixed `n_components`): each probability comes from Beta(b1, b2) or
    Beta(c1, c2) by an indicator with a Beta(d1, d2) weight, updated in the paper's order;
  - `init="asso"`: every chain starts from the Asso factorization (`init_params` may set its
    threshold and weights);
  - `map_components_`, `map_activations_`, `map_log_posterior_`, `map_detection_rate_`,
    `map_background_rate_`: the kept draw with the highest unnormalized log posterior over
    all good chains (fixed `n_components`, no anchors, global rates);
  - `store_draws` accepts an int, the most draws kept per chain (evenly spaced).
  `benchmarks/papers/wagala2026.py` reproduces Table 1, Scenario 1 (regenerated with the
  authors' script; identical to the paper's Figure 2): over three seeds the MAP
  reconstruction scores specificity / F1 / MCC / error rate 0.957 / 0.922 / 0.893 / 0.043 on
  average (error rate 0.038-0.050) against the published 0.960 / 0.928 / 0.903 / 0.039, and
  Asso matches its published row exactly. Scenario 2's data, read from Figure 3, are not the
  data behind Table 1 (Asso and the authors' own R code score differently on it); BBMF still
  beats Asso there (error rate 0.012 against 0.021; published 0.019 against 0.037).
- `BooleanMF`: standard Boolean matrix factorization, `fit` / `fit_transform` / `transform` /
  `inverse_transform` with 0/1 usage and components, `reconstruction_err_` (Hamming distance)
  and `coverage_`. Algorithms:
  - `algorithm="asso"` (Miettinen et al. 2008, The Discrete Basis Problem), with `threshold`
    (tau), `positive_weight` and `negative_weight` (w+ and w-). Reproduces the Digits errors
    of Table 3 with the best tau and w+ over a grid (120,980 / 103,466 / 85,892 at k = 5 / 10
    / 20, published 124,600 / 108,500 / 87,800; the paper tuned but did not list them) and,
    exactly, the Mushroom coverage counts reported for Asso by Belohlavek & Trnecka (2015,
    Table 4: 2 / 6 / 36 factors for 25 / 50 / 75%, never 95%).
  - `algorithm="grecond"` (Belohlavek & Vychodil 2010). Reproduces the factor counts of
    Belohlavek & Trnecka (2015, Table 4) for Mushroom (3 / 7 / 24 / 62 / 120 against 3 / 7 /
    24 / 63 / 120 for 25 / 50 / 75 / 95 / 100%) and Tic-tac-toe (5 / 12 / 19 / 28 / 32,
    exact); Chess is one higher at partial coverage and exact at 124 for the full cover (the
    paper's Chess matrix has one more column).
  - `algorithm="panda"`: PANDA+ (Lucchese et al. 2014), Algorithms 1-3, with `cost` (`"je"`
    the MDL Typed XOR encoding of Miettinen & Vreeken 2011, which the paper cites; `"jp"` with
    `rho`; `"ja"`), `row_tolerance` / `column_tolerance` (eps_r / eps_c), `item_order`
    (frequency, couples, correlation) and `n_rounds` of randomized orders. Reproduces Table 4
    on the paper's synthetic generator (`benchmarks/papers/lucchese2014.py`, 16 cells, the
    thresholds and item order swept as in Table 3). In 14 of 16 cells the number of patterns
    is between K and K + 3 (published: up to K + 5) and J_E is within 0.03 of the embedded
    patterns' own J_E on the same draw and within 0.06 of the published value. The misses are
    at K = 20: at 3% noise 28 patterns and J_E 0.411 (published 23 and 0.41; embedded patterns
    0.357; 25 and 0.398 with 20 randomized rounds), at 7% J_E 0.576 (published 0.60; embedded
    0.546).
  - `n_components=None` stops by itself (GreConD at an exact cover, Asso when nothing improves
    its cover function, PANDA+ when a component would raise its cost), and
    `coverage` stops at a fraction of the ones covered.
  Test data: UCI Mushroom and Tic-tac-toe (via PMLB), bundled in `tests/data/`.
- Split–merge moves (`split_merge=True`, 10 proposals per sweep; an int sets the number, False
  turns them off). Each proposal changes two components at once and is accepted with the exact
  Metropolis–Hastings probability:
  - split, merge and reallocate, proposed by restricted Gibbs scans from a launch state that
    depends only on the union of the two components (Jain & Neal 2004);
  - factor and unfactor, rewrites that keep which entries are covered: shared features move into
    the component whose carriers contain the other's, or back.
  Partners for merge, reallocate, factor and unfactor are drawn by similarity (Jaccard of
  carriers plus Jaccard of members), and the reverse selection probability is computed in the
  proposed state. `split_merge_acceptance_` reports acceptance rates per move type.
- Tests that apply the moves to draws from the exact posterior of small problems (enumerated)
  and check that the draws still follow it.
- Per-sample rates: `detection_effects=("sample",)` and `background_effects=("sample",)` (the
  spec's parameters). Each sample's rate is logit-normal around the population rate, whose
  prior is `detection_prior` / `background_prior`; the spread has a half-Cauchy(0, 1) prior.
  All updates are slice sampling; under `or_flip` each sample keeps detection above
  background. New attributes `detection_rate_per_sample_`, `background_rate_per_sample_`, their
  `*_interval_` (95%) and `detection_spread_` / `background_spread_`. New samples in
  `transform` and `score` use the population rates. `make_boolean_factors` accepts per-sample
  `detection` and `background` arrays.
  On a simulation with per-sample rates (300 samples, 6 components, logit spreads 1.0 and 0.7),
  with either likelihood all 4 chains converge (without effects: 1 of 4 under `or_flip`, 0 of 4
  under `noisy_or`), held-out log density improves from -0.102 to -0.098, the spreads are
  recovered (1.10 and 0.68, true 1.0 and 0.7) and 95% intervals cover 94-95% of the true rates.
  Under `or_flip` a spurious seventh component also disappears.
  Per-feature rates are not implemented yet (`NotImplementedError`).
- Per-component detection rates with `likelihood="noisy_or"`:
  `detection_effects=("component",)`, alone or with `"sample"`. Each active component delivers
  each of its features independently with its own rate, logit-normal around the population rate
  with a learned spread; with both levels logit lambda_ik = y_i + g_k. Asking for component rates
  with `or_flip` raises `ValueError`: there a covered entry has a single detection rate however
  many components cover it, so no component's own rate can be identified. New attributes
  `detection_rate_per_component_`, its `_interval_` (95%) and `detection_component_spread_`.
  Inference keeps L[i, j] = sum of log(1 - lambda_ik) over covering components and updates one
  component's rate at a time by slice sampling on the entries it covers; an exact move along the
  direction the likelihood cannot see (population rate and sample offsets up, component offsets
  down) keeps the two levels from drifting. Split–merge moves, `transform` and
  `inverse_transform` use the per-component rates. `make_boolean_factors` gains
  `component_detection` to simulate this model.

- Option keywords for published methods, and `boolmf.presets` of named keyword sets that
  reproduce them (`BayesianBooleanMF(n_components=L, **presets.rukat2017)`):
  - `activation_prior` / `activation_level` and `membership_level` (with `membership_prior`
    now also accepting a float or `"empirical"`): Beta rates per component, shared by all
    components, or per sample / per feature, or a fixed rate;
  - `alpha_prior` accepts a float to fix alpha;
  - `tied_rates` (`or_flip` with background = 1 - detection) and `rate_estimation="mle"`
    (maximum-likelihood rates after every sweep);
  - `update="metropolised"` (Metropolised Gibbs flips, Liu 1996) and `update_order`;
  - `init="uniform"` and `init="empty"`.
- `presets.rukat2017`, the OrMachine (Rukat et al. 2017). `benchmarks/papers/rukat2017.py`
  reproduces Section 4.1: all 17 points read from Fig. 4 are within 0.03 of the published
  reconstruction errors; a fast version runs in the test suite.
- Collapsed Indian buffet process samplers, `births="enumerate"` (Wood, Griffiths & Ghahramani
  2006: each row keeps an existing component with prior m / N, then draws its number of new
  components, up to `max_births`, from the exact conditional with their other side summed out)
  and `births="metropolis"` (Meeds et al. 2007: a Poisson(alpha / N) number of new components,
  drawn from the prior, replaces the row's own ones with Metropolis–Hastings acceptance).
  `birth_members="gibbs"` sets the new components' other side by one Gibbs pass from zero, as
  the papers write it, instead of the exact joint draw. `ibp_side="features"` puts the Indian
  buffet process on the features. Tests check both birth moves against the exact posterior of
  a problem small enough to enumerate. With these samplers `max_components="auto"` is
  `min(n_samples, n_features) + 2 * max_births`, and a warning is raised if a birth finds no
  free slot.
- `presets.wood2006` (Wood et al. 2006) and `presets.rukat_yau2019` (Rukat & Yau 2019).
  `benchmarks/papers/wood2006.py` reproduces Fig. 4: over 30 datasets per structure, 13 of 16
  read-off points are within tolerance. At 1000 iterations the degree-1, undercomplete and
  overcomplete graphs match (for example overcomplete in-degree error 2.78, published 2.8, and
  structure error 5.7, published 3.8 +- 2.0); the disconnected graph reaches 0.39 and 0.21
  against 0.05 and 0.05, because a few datasets keep a duplicated cause with its trials split
  between the copies. With `births="metropolis"` errors stay near those of the paper's RJMCMC
  sampler: births proposed from the prior are rarely accepted.
  `benchmarks/papers/rukat_yau2019.py` reproduces Section 3.1 in part: the posterior mode of the
  number of codes equals the true rank (2-10) in 74%, 67% and 59% of runs at 0%, 10% and 20%
  flips (mean excess +0.22, +0.44, +0.63); from rank 5 up it is almost always exact, but at
  ranks 2-4 a run often keeps an extra code that splits a dense true code. The presets'
  docstrings list their departures from the papers, and where the authors' public code
  departs from its paper.
- `redundant_components_` and a warning when robust components look redundant: two
  components with nearly the same carriers or members (Jaccard >= 0.9), or a component whose
  entries are at least 90% covered by the others. Under the OR model such components add almost
  nothing to the fit and usually mark a component split in two or duplicated. On the
  Rukat & Yau (2019) reproduction grid (81 fits) it fires in 14 of the 26 fits that overestimate
  the rank and in none of the 54 exact ones; it does not fire on the nested-factors benchmark.
- `benchmarks/papers/rukat_yau2019.py` compares with the authors' code on the same data: it is
  exact in 64 of 81 runs (this preset: 54), merges codes at ranks 2-3, and never shows the
  published overestimate at 20% flips.

### Changed
- `component_leakage` (and so `summary()`) no longer keeps one samples x features array per
  component; on a 500 x 4,811 pangenome fit with 430 components it needed 8.3 GB and ran out of
  memory. Results are unchanged.
- `BayesianBooleanMF.fit_transform` now returns `fit(X).transform(X)`, as scikit-learn's
  transformer contract expects; the activation probabilities from the chains themselves stay
  in `activations_`.
- The Bayesian estimator is now `BayesianBooleanMF` (it was `BoolMF`, alias `BooleanMF`), and
  `BooleanMF` is the new standard estimator, following scikit-learn's pairs such as
  `GaussianMixture` / `BayesianGaussianMixture`. The module `boolmf._estimator` is now
  `boolmf._bayesian`. `get_feature_names_out` returns `bayesianbooleanmf0`, ... .
- With a fixed `n_components`, every slot in use is reported as a component, and the slot
  resets used for births under the Indian buffet process no longer apply.
- Nested-factors benchmark (500 samples, 20 true components, 8 chains; seeds 0 and 1), before →
  after, `or_flip`:
  - chains flagged `stuck`: 1 → 0 on both seeds;
  - cross-chain R-hat of the detection rate: 1.33 → 1.004 and 1.005;
  - held-out perplexity gap to the true model: 0.0084 → 0.0056 (seed 0), 0.0082 → 0.0057
    (seed 1); calibration: 0.995 and 0.996 → 1.000;
  - overlapping components recovered: 10/12 and 10/12 → 10/12 and 11/12. The component that lives in 60%
    of a small group, which 7 of 8 chains used to fold into its group, is now found by all
    chains. The remaining misses are the nested and partial components, which the chains write
    in an equivalent form (same covered entries, features attached to the component
    whose carriers always carry them);
  - run time: 582 and 590 s → 482 and 462 s (fewer spurious components to update).
  Still open: R-hat of the background rate, alpha and the log-likelihood reaches 1.17, 1.17 and
  1.8 on seed 1, because chains differ in how many small components absorb noise.
- The default likelihood stays `"or_flip"`. With these moves `"noisy_or"` recovers the same
  components with the same fit, but the number of supported components mixes worse across chains
  (R-hat 2.1 and 1.9 against 1.01 and 1.02).
- Version 0.2.0.dev0.

## 0.1.0 (unreleased)

First release.

Supports Python 3.12, 3.13 and 3.14. Minimum dependency versions are the oldest with wheels for
Python 3.12 (NumPy 1.26, SciPy 1.11.2, numba 0.59, scikit-learn 1.6) and, for joblib, the oldest
that runs on Python 3.14 (1.5); the test suite passes at these minimums.

### Added
- `BoolMF` estimator (alias `BooleanMF`): Bayesian Boolean matrix factorization with an Indian
  buffet process prior (`n_components=None`) or a fixed number of components.
- Two likelihoods: `"or_flip"` (default) and `"noisy_or"`.
- Anchor components (`AnchorComponent`) active in every sample, with fixed or learned members.
- Missing data through NaN or `mask=`; sparse and pandas input.
- Multi-chain Gibbs sampling with numba kernels (parallel over features and samples), chains in
  parallel with joblib, counter-based random streams (results independent of thread count).
- Automatic burn-in (segment R-hat on the component count and rates plus a Geweke check on the
  log-likelihood), automatic thinning, chain status (`ok`, `not_converged`, `stuck`).
- Cross-chain Hungarian matching, robustness scores and component flags.
- `transform` (projection of new samples, row-order invariant), `inverse_transform`, `score`,
  `score_samples`, `explained_probability`, `predictive_probability`, `binarize_components`
  (threshold or Bayesian FDR), `get_draws`, `summary`, `get_feature_names_out`.
- `boolmf.diagnostics` (rank-normalized split R-hat, ESS, Geweke), `boolmf.matching`,
  `boolmf.metrics` (confusion tables overall and per sample or feature, integrity, leakage,
  calibration), `boolmf.model_selection` (`EntryKFold`, `EntryShuffleSplit`,
  `cross_validate_entries`), `boolmf.datasets` (`make_boolean_factors`, `make_nested_factors`).
- Passes scikit-learn's `parametrize_with_checks` with no expected failures.

### Notes and deviations from the design spec
- The default likelihood is `"or_flip"`. Under `"noisy_or"`, a component whose members are split
  across two components with the same carriers is a local mode that one-at-a-time updates
  rarely leave; split–merge moves (v0.2) are planned before reconsidering the default.
- The binarization method is `binarize_components` because `binarize` is a constructor
  parameter (input binarization threshold, as in `BernoulliNB`).
- Burn-in is decided per chain; cross-chain R-hat is reported after sampling in `rhat_`.
- Convergence warnings use the detection and background rates (R-hat above
  `max(1.05, rhat_threshold)`, or pooled ESS below `min_ess`). The log-likelihood and the
  component count are reported in `rhat_` and `ess_` but do not trigger warnings: they mix
  slowly because short-lived components that absorb noise (one carrier, or one member) come and
  go. On the quickstart simulation the number of entries they cover has correlation 0.91 with
  the log-likelihood, while every robust component stays fixed across draws.
- Not yet implemented (roadmap): per-sample, per-feature and per-component rate effects,
  split–merge moves, weighted population summaries, ArviZ export.
