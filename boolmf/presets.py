"""Named presets that reproduce published methods with ``BoolMF``.

Each preset is a read-only mapping of keyword arguments::

    from boolmf import BoolMF, presets
    model = BoolMF(n_components=7, **presets.rukat2017)

Passing a keyword that the preset also sets raises ``TypeError``; to change one, merge first:
``BoolMF(n_components=7, **{**presets.rukat2017, "n_chains": 4})``. ``benchmarks/papers/``
reproduces each paper's published results with its preset, and each preset's docstring lists
where the implementation departs from the paper and why.
"""

from types import MappingProxyType

__all__ = ["rukat2017", "PRESETS"]

rukat2017 = MappingProxyType(dict(
    likelihood="or_flip",
    tied_rates=True,
    rate_estimation="mle",
    activation_prior="empirical",
    activation_level="shared",
    membership_prior="empirical",
    membership_level="shared",
    update="metropolised",
    update_order="activations_first",
    split_merge=False,
    init="uniform",
    n_chains=1,
    burn_in=100,
    n_draws=100,
    thin=1,
))
"""The OrMachine of Rukat, Holmes, Titsias & Yau (2017), Bayesian Boolean Matrix
Factorisation, ICML (PMLR 70: 2969-2978).

Pass ``n_components`` (the paper's fixed L). Model: x = OR(z AND u) with symmetric flip noise,
P(correct) = sigmoid(lambda) set to its maximum-likelihood value after every sweep; z and u
independent Bernoulli with a fixed probability chosen from the data density
("empirical Bayes", p = sqrt(1 - (1 - density)^(1/L))). Sampler: Metropolised Gibbs, Z then
U then lambda; uniform random start; 100 burn-in sweeps and 100 kept samples, one chain.

Departures: none in the model or sampler. ``"empirical"`` takes the density of the observed
(noisy) matrix; in the paper's simulations the prior evidently came from the design density
(see ``benchmarks/papers/rukat2017.py``), so pass that value explicitly to match them. Summaries
(``components_``, ``activations_``) are posterior means, as in the paper.

Reproduced: Section 4.1 (Fig. 4), all 17 read-off points (100 x 100 rank 7 at densities 0.5 and
0.7, 20-50% flips; 1000 x 1000 rank 5) within 0.03 of the published errors.
"""

PRESETS = {"rukat2017": rukat2017}
