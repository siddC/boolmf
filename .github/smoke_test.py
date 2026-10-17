"""Smoke test of an installed boolmf (CI runs it on the built wheel, outside the checkout).

Checks that boolmf is imported from site-packages, that every module in the package imports
(a module left out of the wheel fails here), and that both estimators fit a tiny matrix.
"""

import importlib
import pathlib
import pkgutil

import numpy as np

import boolmf
from boolmf import BayesianBooleanMF, BooleanMF

path = pathlib.Path(boolmf.__file__)
assert "site-packages" in path.parts, f"boolmf imported from {path}, not from the install"
for mod in pkgutil.walk_packages(boolmf.__path__, "boolmf."):
    importlib.import_module(mod.name)

rng = np.random.default_rng(0)
X = (rng.random((30, 20)) < 0.3).astype(float)
BooleanMF(n_components=2).fit(X)
BayesianBooleanMF(n_chains=1, max_sweeps=20, burn_in=10, n_draws=5, random_state=0).fit(X)
print(f"boolmf {boolmf.__version__} from {path.parent}: imports and fits OK")
