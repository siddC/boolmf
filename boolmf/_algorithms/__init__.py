"""Standard (non-Bayesian) Boolean matrix factorization algorithms used by ``BooleanMF``.

Every function takes a boolean matrix X (samples x features) and returns boolean usage
(samples x k) and basis (k x features) matrices with X approximately equal to their Boolean
product.
"""

from .asso import asso, asso_usage
from .grecond import grecond, grecond_usage
from .mebf import mebf

__all__ = ["asso", "asso_usage", "grecond", "grecond_usage", "mebf"]
