# Changelog

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
