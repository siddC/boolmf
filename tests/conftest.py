import pytest

from boolmf import BoolMF
from boolmf.datasets import make_boolean_factors

FAST = dict(n_chains=2, max_sweeps=400, burn_in=200, n_draws=40, thin=2, random_state=0)


@pytest.fixture(scope="session")
def small_data():
    X, truth = make_boolean_factors(120, 90, 3, prevalence=(0.25, 0.45), membership=(0.15, 0.3),
                                    random_state=0, return_truth=True)
    return X, truth


@pytest.fixture(scope="session")
def fitted(small_data):
    X, _ = small_data
    return BoolMF(**FAST).fit(X)
