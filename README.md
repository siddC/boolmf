# boolmf

Boolean matrix factorization for Python, with a scikit-learn API.

`boolmf` factorizes a binary matrix into overlapping latent components. Each component is a set
of features that is either active or inactive in each sample, and a sample's observed features
are the union of its active components (plus noise). Two estimators:

- **`BooleanMF`**: standard algorithms that search for the two binary factor matrices
  (`algorithm="asso"`, Miettinen et al. 2008; `algorithm="grecond"`, Belohlavek & Vychodil
  2010; `algorithm="mebf"`, Wan et al. 2020). Fast and deterministic, with a fixed number of
  components or a coverage target.
- **`BayesianBooleanMF`**: a Bayesian model sampled by MCMC. The number of components is
  learned (Indian buffet process prior), membership and activation come back as posterior
  probabilities, noise rates are estimated, and missing entries are allowed.

```python
from boolmf import BooleanMF

model = BooleanMF(n_components=10, algorithm="asso", threshold=0.8)
W = model.fit_transform(X)          # X: samples x features, 0/1; W: usage, 0/1
model.components_                   # components x features, 0/1
model.reconstruction_err_           # entries where the Boolean product of W and H differs from X
BooleanMF(algorithm="grecond").fit(X)   # exact cover from below, components never cover a 0
```

```python
from boolmf import BayesianBooleanMF

model = BayesianBooleanMF(n_chains=20, n_jobs=-1, random_state=0).fit(X)
model.components_        # P(feature j is a member of component k), shape (K, n_features)
model.activations_       # P(component k is active in sample i),    shape (n_samples, K)
model.n_components_      # number of robust components
model.summary()          # robustness, prevalence, support, integrity, leakage per component
members, active = model.binarize_components()          # Bayes rule at 0.5
members, active = model.binarize_components(method="bfdr", fdr=0.01)
model.transform(X_new)   # activation probabilities for new samples
```

## The Bayesian model

For samples *i*, features *j* and components *k*, let *z<sub>ik</sub>* = 1 when component *k*
is active in sample *i* and *u<sub>jk</sub>* = 1 when feature *j* is a member of component *k*.
With *c<sub>ij</sub>* the number of active components containing feature *j* in sample *i*:

| `likelihood` | P(x<sub>ij</sub> = 1) |
| --- | --- |
| `"or_flip"` (default) | detection rate if *c<sub>ij</sub>* ≥ 1, background rate otherwise |
| `"noisy_or"` | 1 − (1 − background)(1 − detection)<sup>*c<sub>ij</sub>*</sup> |

Priors: *z<sub>ik</sub>* ~ Bernoulli(π<sub>k</sub>) with π<sub>k</sub> ~ Beta(α/K, 1) (a truncated
Indian buffet process, α ~ Gamma(1, 1)); *u<sub>jk</sub>* ~ Bernoulli(ρ<sub>k</sub>) with
ρ<sub>k</sub> ~ Beta(1, 1); Beta(1, 1) priors on both rates by default. Pass `n_components=K` for
a fixed number of components.

Rates can vary by sample: `detection_effects=("sample",)` and `background_effects=("sample",)`
give each sample its own rate, logit-normal around a population rate with a learned spread
(`detection_rate_per_sample_`, with 95% intervals). A sample with a low detection rate points to
an incomplete profile; one with a high background rate to many features no component explains.
With `likelihood="noisy_or"`, `detection_effects=("component",)` gives each component its own
detection rate (`detection_rate_per_component_`): how completely its features show up in its
carriers.

Inference runs several independent Gibbs chains (numba, parallel within and across chains).
Each sweep adds Metropolis–Hastings moves on pairs of components (split, merge, reallocate, and
rewrites that keep the fit), so chains can leave states that one-variable-at-a-time updates
cannot. Components are matched across chains with the Hungarian algorithm on Jaccard
similarity, and the draws are pooled. Components are flagged `robust`, `low_support` or `not_robust`; convergence is
reported with rank-normalized split R-hat and effective sample size (`rhat_`, `ess_`).

## Published methods

Every method is checked by reproducing its paper's published results on the same inputs
(`benchmarks/papers/`). For `BooleanMF`, the `algorithm` keyword selects the method: Asso
reproduces the Digits errors of Miettinen et al. (2008, Table 3) within 5% and the Mushroom
coverage counts of Belohlavek & Trnecka (2015, Table 4) exactly; GreConD reproduces the
Mushroom, Tic-tac-toe and Chess counts of the same table. MEBF follows its paper's pseudocode;
its published results came from different code and are not reproduced (see
`benchmarks/papers/wan2020.py`).

For `BayesianBooleanMF`, `boolmf.presets` holds keyword sets that reproduce published
samplers:

```python
from boolmf import BayesianBooleanMF, presets
BayesianBooleanMF(n_components=7, **presets.rukat2017)     # OrMachine, Rukat et al. (2017)
BayesianBooleanMF(**presets.rukat_yau2019)                 # OrMachine with an IBP, Rukat & Yau (2019)
BayesianBooleanMF(**presets.wood2006)                      # hidden causes, Wood et al. (2006)
```

The same ingredients are available one by one (`tied_rates`, `rate_estimation`, `update`,
`activation_prior`, `membership_level`, ...). Under the Indian buffet process, `births` picks
how components appear and disappear: `"slots"` (the default, a truncated pool of component
probabilities, used with split–merge moves), `"enumerate"` (the collapsed Gibbs sampler of
Wood et al. 2006, which draws each row's number of new components from its exact conditional)
or `"metropolis"` (the Metropolis–Hastings births of Meeds et al. 2007). `ibp_side="features"`
puts the buffet on the features, as Wood et al. do. Each preset's docstring lists where it
departs from the paper and how closely the published results are reproduced.

## Validation

```python
from boolmf.model_selection import EntryKFold, cross_validate_entries
from boolmf.metrics import confusion_table, calibration_curve

cross_validate_entries(BayesianBooleanMF(), X, cv=EntryKFold(5))   # hold out entries, score them
confusion_table(model, level="sample")                  # precision, recall, specificity, NPV
```

## Installation

```bash
pip install boolmf            # once released
pip install git+https://github.com/siddC/boolmf
```

Requires Python 3.12 or later, NumPy, SciPy, scikit-learn 1.6+, numba and joblib. No compiler is needed.

## Status

Version 0.1 (alpha). `CHANGELOG.md` lists what is implemented and `ROADMAP.md` what comes next.

## References

- Wood, Griffiths & Ghahramani (2006). A non-parametric Bayesian method for inferring hidden causes. UAI.
- Rukat, Holmes, Titsias & Yau (2017). Bayesian Boolean matrix factorisation. ICML.
- Jain & Neal (2004). A split-merge Markov chain Monte Carlo procedure for the Dirichlet process mixture model. Journal of Computational and Graphical Statistics.
- Griffiths & Ghahramani (2011). The Indian buffet process: an introduction and review. JMLR.
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021). Rank-normalization, folding, and localization: an improved R-hat. Bayesian Analysis.
- Barbieri & Berger (2004). Optimal predictive model selection. Annals of Statistics.
- Newton, Noueiry, Sarkar & Ahlquist (2004). Detecting differential gene expression with a semiparametric hierarchical mixture method. Biostatistics.

## License

BSD 3-Clause.
