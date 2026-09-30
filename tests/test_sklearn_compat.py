"""scikit-learn estimator checks (all pass; no expected failures).

The checks feed continuous random data, so they run with ``binarize=0.5`` and small settings.
"""

import warnings

from sklearn.utils.estimator_checks import parametrize_with_checks

from boolmf import BayesianBooleanMF, BooleanMF

SHORT = dict(n_chains=2, max_sweeps=40, burn_in=20, n_draws=10, thin=1, random_state=0)


@parametrize_with_checks(
    [BooleanMF(binarize=0.5),
     BooleanMF(3, algorithm="grecond", binarize=0.5),
     BooleanMF(3, algorithm="mebf", binarize=0.5),
     BayesianBooleanMF(binarize=0.5, **SHORT),
     BayesianBooleanMF(binarize=0.5, detection_effects=("sample",),
                       background_effects=("sample",), **SHORT),
     BayesianBooleanMF(binarize=0.5, likelihood="noisy_or",
                       detection_effects=("sample", "component"),
                       background_effects=("sample",), **SHORT)],
)
def test_sklearn_compatible_estimator(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check(estimator)
