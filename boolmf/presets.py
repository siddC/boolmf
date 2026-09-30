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

__all__ = ["rukat2017", "rukat_yau2019", "wood2006", "PRESETS"]

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

wood2006 = MappingProxyType(dict(
    likelihood="noisy_or",
    ibp_side="features",
    births="enumerate",
    max_births=10,
    birth_members="gibbs",
    activation_level="shared",
    split_merge=False,
    init="empty",
    n_chains=1,
))
"""The infinite noisy-OR hidden-cause model of Wood, Griffiths & Ghahramani (2006), A
Non-Parametric Bayesian Method for Inferring Hidden Causes, UAI (arXiv:1206.6865).

Samples are trials and features are observed variables. The Indian buffet process is on the
links from hidden causes to observed variables (``ibp_side="features"``); each cause is active
in a trial with probability p shared by all causes. P(x = 1) = 1 - (1 - epsilon)(1 - lambda)^c.
Priors as in the paper's stroke data analysis: Beta(1, 1) on lambda, epsilon and p, Gamma(1, 1)
on alpha (pass floats to fix them, as in the paper's simulations). Sampler: the collapsed Gibbs
sampler of Algorithm 2, one observed variable at a time, keeping each existing cause with prior
m / N and then drawing the number of new causes from its exact conditional, up to 10; then the
activations; one chain from an empty Z. The run length is not part of the preset.

The new causes' activations start at 0 and get one Gibbs pass (``birth_members="gibbs"``),
as in Algorithm 2. With two or more new causes this pass is not a draw from their joint
conditional, so the chain does not target the exact posterior; ``birth_members="exact"``
does.

Departures: lambda and epsilon are updated by slice sampling instead of
Metropolis steps, and p and alpha by the same Gibbs steps; both leave the same posterior
invariant. Summaries are posterior means over the draws.

Reproduced: Section 5, Fig. 4 (four structures, 30 datasets each; see
``benchmarks/papers/wood2006.py``). 13 of the 16 read-off points are within tolerance, including
all four structures' in-degree and structure errors at 1000 iterations except the disconnected
graph (0.39 and 0.21 against 0.05 and 0.05: a few datasets keep a duplicated cause with its
trials split between the copies for hundreds of sweeps). The other two misses are at 10
iterations, where the published standard deviations are 0.05 and 0.4.
"""

rukat_yau2019 = MappingProxyType(dict(
    likelihood="or_flip",
    tied_rates=True,
    rate_estimation="mle",
    births="enumerate",
    max_births=9,
    birth_members="gibbs",
    membership_prior=0.5,
    membership_level="shared",
    alpha_prior=3.0,
    split_merge=False,
    init="empty",
    n_chains=1,
    burn_in=100,
    n_draws=100,
    thin=1,
))
"""The OrMachine with an Indian buffet process prior, Rukat & Yau (2019), Bayesian
Nonparametric Boolean Factor Models (arXiv:1907.00063).

Model: the OrMachine likelihood (symmetric flip noise, rate set to its maximum-likelihood value
after every sweep) with Z ~ IBP(alpha) on the samples and U independent Bernoulli(q). Sampler:
Algorithm 1, rows of Z in turn: existing codes with prior m / N, then the number of new codes
L' < 10 from Eqs. 6-8; then U; 200 samples, the first 100 discarded, one chain.

The paper does not give q, alpha or the start: the preset takes q = 0.5 and alpha = 3 from the
authors' code (the ``lom`` package) and starts from an empty Z. New codes' memberships start at
0 and get one Gibbs pass, as in Algorithm 1 (``birth_members="gibbs"``).

Departures: Algorithm 1 removes a row's own codes while it passes over the other codes; here
they are removed after the pass, as in Wood et al. (2006), since removing them earlier makes the
later updates condition on a state without them. The authors' code departs from the paper in
ways the preset does not follow: it considers at most 2 new codes, gives L' a prior
proportional to (alpha / N)^(L + L') / (L + L')! with L the current number of codes, and
ignores q when sampling new codes.

Reproduced in part: Section 3.1 (see ``benchmarks/papers/rukat_yau2019.py``; 3 datasets per
rank 2-10). The posterior mode equals the true rank in 74%, 67% and 59% of runs at 0%, 10% and
20% flips, with mean excess +0.22, +0.44 and +0.63: overestimation grows with noise as
published, but less than the one extra code reported at 20%, and at ranks 5-10 the rank is
almost always exact. At ranks 2-4 (dense codes, factor density 0.45-0.54) a run often keeps one
or two extra codes that split a true code, a local mode that one-at-a-time updates leave slowly,
so "reliably recovers" holds only from rank 5 up.
"""

PRESETS = {"rukat2017": rukat2017, "rukat_yau2019": rukat_yau2019, "wood2006": wood2006}
