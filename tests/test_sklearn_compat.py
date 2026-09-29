"""scikit-learn estimator checks (all pass; no expected failures).

The checks feed continuous random data, so they run with ``binarize=0.5`` and small settings.
"""

import warnings

from sklearn.utils.estimator_checks import parametrize_with_checks

from boolmf import BoolMF


@parametrize_with_checks(
    [BoolMF(binarize=0.5, n_chains=2, max_sweeps=40, burn_in=20, n_draws=10, thin=1,
            random_state=0),
     BoolMF(binarize=0.5, detection_effects=("sample",), background_effects=("sample",),
            n_chains=2, max_sweeps=40, burn_in=20, n_draws=10, thin=1, random_state=0),
     BoolMF(binarize=0.5, likelihood="noisy_or", detection_effects=("sample", "component"),
            background_effects=("sample",), n_chains=2, max_sweeps=40, burn_in=20, n_draws=10,
            thin=1, random_state=0)],
)
def test_sklearn_compatible_estimator(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check(estimator)
