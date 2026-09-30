"""Fit BayesianBooleanMF to synthetic data and inspect the result (needs pandas for the summary)."""
import numpy as np

from boolmf import BayesianBooleanMF
from boolmf.datasets import make_boolean_factors
from boolmf.matching import jaccard_matrix
from boolmf.metrics import confusion_table

X, truth = make_boolean_factors(n_samples=300, n_features=400, n_components=6, missing=0.05,
                                random_state=0, return_truth=True)

model = BayesianBooleanMF(n_chains=4, n_jobs=-1, max_sweeps=3000, random_state=0).fit(X)
print(model.summary().round(3).to_string())
print("robust components:", model.n_components_)
print(f"detection rate {model.detection_rate_:.3f}, "
      f"background rate {model.background_rate_:.4f}")

members, active = model.binarize_components()
robust = model.component_flags_ == "robust"
J = jaccard_matrix(truth["members"], members[robust])
print("best Jaccard to each true component:", np.round(J.max(axis=1), 2))
print(confusion_table(model))
