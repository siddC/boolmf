"""boolmf: Boolean matrix factorization.

A binary matrix X (samples x features) is modelled as the Boolean product of two binary
matrices: components (sets of features) and their usage by each sample.

* ``BooleanMF``: standard algorithms (Asso, GreConD) that search for the factor matrices.
* ``BayesianBooleanMF``: a Bayesian model with a noisy-OR likelihood and an Indian buffet
  process prior on the number of components, sampled by MCMC.
"""

from . import presets
from ._bayesian import AnchorComponent, BayesianBooleanMF, BetaMixture
from ._boolean import BooleanMF

__version__ = "0.2.0.dev0"

__all__ = ["BooleanMF", "BayesianBooleanMF", "AnchorComponent", "BetaMixture", "presets",
           "__version__"]
