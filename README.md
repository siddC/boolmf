# boolmf

Bayesian nonparametric Boolean matrix factorization for Python.

`boolmf` factorizes a binary matrix into overlapping latent components. Each component is a set
of features that is either active or inactive in each sample, and a sample's observed features
are the union of its active components plus noise. The number of components is learned from the
data (Indian buffet process prior), membership and activation come back as posterior
probabilities, and the API follows scikit-learn.

```python
from boolmf import BoolMF

model = BoolMF(n_chains=20, n_jobs=-1, random_state=0).fit(X)   # X: samples x features, 0/1
model.components_        # P(feature j is a member of component k), shape (K, n_features)
model.activations_       # P(component k is active in sample i),    shape (n_samples, K)
model.n_components_      # number of robust components
model.summary()          # robustness, prevalence, support, integrity, leakage per component
members, active = model.binarize_components()          # Bayes rule at 0.5
members, active = model.binarize_components(method="bfdr", fdr=0.01)
model.transform(X_new)   # activation probabilities for new samples
```

## The model

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

Inference runs several independent Gibbs chains (numba, parallel within and across chains),
matches components across chains with the Hungarian algorithm on Jaccard similarity, and pools
the draws. Components are flagged `robust`, `low_support` or `not_robust`; convergence is
reported with rank-normalized split R-hat and effective sample size (`rhat_`, `ess_`).

## Validation

```python
from boolmf.model_selection import EntryKFold, cross_validate_entries
from boolmf.metrics import confusion_table, calibration_curve

cross_validate_entries(BoolMF(), X, cv=EntryKFold(5))   # hold out entries, score them
confusion_table(model, level="sample")                  # precision, recall, specificity, NPV
```

## Installation

```bash
pip install boolmf            # once released
pip install git+https://github.com/siddC/boolmf
```

Requires Python 3.10+, NumPy, SciPy, scikit-learn 1.6+, numba and joblib. No compiler is needed.

## Status

Version 0.1 (alpha). See `CHANGELOG.md` for what is implemented and the roadmap for per-sample,
per-feature and per-component rates, split–merge moves and weighted population summaries.

## References

- Wood, Griffiths & Ghahramani (2006). A non-parametric Bayesian method for inferring hidden causes. UAI.
- Rukat, Holmes, Titsias & Yau (2017). Bayesian Boolean matrix factorisation. ICML.
- Griffiths & Ghahramani (2011). The Indian buffet process: an introduction and review. JMLR.
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021). Rank-normalization, folding, and localization: an improved R-hat. Bayesian Analysis.
- Barbieri & Berger (2004). Optimal predictive model selection. Annals of Statistics.
- Newton, Noueiry, Sarkar & Ahlquist (2004). Detecting differential gene expression with a semiparametric hierarchical mixture method. Biostatistics.

## License

BSD 3-Clause.
