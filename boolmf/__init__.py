"""boolmf: Bayesian nonparametric Boolean matrix factorization.

A binary matrix X (samples x features) is modelled as the union of latent components,
each with binary members and active or inactive in each sample, under a noisy-OR
likelihood and an Indian buffet process prior on the number of components.
"""

from . import presets
from ._estimator import AnchorComponent, BooleanMF, BoolMF

__version__ = "0.2.0.dev0"

__all__ = ["BoolMF", "BooleanMF", "AnchorComponent", "presets", "__version__"]
