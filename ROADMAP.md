# Roadmap

Three releases, each closed by a gate that must pass before the next one starts. No dates are
set.

## v0.1: core estimator (done)

- `or_flip` and `noisy_or` likelihoods; Indian buffet process prior with slot birth and death
- Missing data and masking; multi-chain MCMC with automatic burn-in and thinning
- Rank-normalized R-hat, ESS, Hungarian matching across chains, robustness flags
- `transform`, `score`, `binarize_components`, NMF initialization
- Metrics (confusion tables, integrity, leakage, calibration) and entry-wise cross-validation

**Gate 1 (passed):** on the nested-factors benchmark, 8/8 exclusive groups and at least 9/12
overlapping components recovered, calibration of at least 99%, and all scikit-learn estimator
checks pass.

## v0.2: real data

- Split–merge moves, so chains stop settling in different modes (done)
- Per-sample detection and background rates (done)
- Per-component detection rates, with `likelihood="noisy_or"` (done)
- Coarsened posterior, `likelihood_power` (Miller & Dunson 2019), so the number of components
  stops growing with the data when the data are not exactly Boolean (done)
- The original samplers of the cited papers as options, each checked by reproducing the
  paper's published results: Rukat et al. 2017 (done), Wood et al. 2006, Meeds et al. 2007
  births and Rukat & Yau 2019 (done), Wagala et al. 2026 (done)
- Standard Boolean matrix factorization as its own estimator, `BooleanMF`, next to
  `BayesianBooleanMF`: Asso, GreConD and PANDA+ (done). MEBF
  (Wan et al. 2020) was dropped: its published results come from code that differs from the
  paper in five places, and the paper's own algorithm stalls on noisy data
- First real dataset: a subsample of a bacterial pangenome presence/absence matrix, compared with
  an NMF baseline; binarization at 0.5 against Bayesian FDR

**Gate 2:** fits to the real matrix converge (R-hat of at most 1.01 across chains) and score at
least as well as the NMF baseline on held-out entries.

Status (GENOMiCUS-4k, 200 and 500 genomes, `benchmarks/genomicus4k.py`): held-out scores pass
(log-likelihood -0.115 against NMF's best -0.148 at 500 genomes; -0.123 against -0.199 at 200).
Convergence fails: R-hat of the rates 1.07-2.8 depending on the variant, of the log-likelihood
about 3, because the chains keep adding components (hundreds) after 4,000 sweeps. On a
phylogroup-balanced subsample (249 genomes) 4 chains of 20,000 sweeps still have not settled
(log-likelihood rising, about 610 components, R-hat 2.8), and chains agree on prediction but
not on components (14% matched at Jaccard >= 0.8). What the gate should check is open.

After v0.2, CI adds `macos-latest` and `windows-latest` for every supported Python version.

## v0.3: scale

- Per-feature rates with shrinkage
- Weighted population summaries (prevalence corrected for uneven sampling)
- ArviZ export
- Full-size fits (thousands of samples)
- Documentation site and PyPI release

**Gate 3 (release 1.0):** full-size fits converge, documentation and examples are complete, and
the API is frozen.

After v0.3, CI adds Python 3.15.

## Research track

Alongside the releases: latent sample clusters, rare features, and how to treat phantom and
low-support components.
