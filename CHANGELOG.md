# Changelog

## 0.2.0 (unreleased)

### Added
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

### Changed
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
