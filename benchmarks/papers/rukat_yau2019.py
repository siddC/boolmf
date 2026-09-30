"""Reproduce Rukat & Yau (2019), Section 3.1: the number of latent dimensions.

200 x 500 matrices of balanced density are Boolean products of random factors of rank L = 2..10
(each factor entry is 1 with probability sqrt(1 - 0.5^(1/L)), so the product has density 0.5),
optionally with 10% or 20% of bits flipped. The sampler (``presets.rukat_yau2019``) draws 200
samples and discards the first 100. The paper shows no figures, only these claims:

* without noise, the model reliably recovers the true number of latent dimensions;
* with 10% noise, recovery is close to perfect;
* with 20% noise, the number is systematically overestimated by about one.

The number of latent dimensions of a sample is the number of codes with at least one carrier
and one member; the posterior mode over the kept samples is compared with L.

Run: python benchmarks/papers/rukat_yau2019.py [--repeats 3]
"""
import argparse
import warnings

import numpy as np

from boolmf import BoolMF, presets

N, D, DENSITY = 200, 500, 0.5
RANKS = range(2, 11)
NOISE = (0.0, 0.1, 0.2)


def simulate(L, flip, rng):
    p = np.sqrt(1.0 - (1.0 - DENSITY) ** (1.0 / L))
    Z = rng.random((N, L)) < p
    U = rng.random((D, L)) < p
    truth = (Z.astype(int) @ U.T.astype(int)) > 0
    return np.where(rng.random(truth.shape) < flip, ~truth, truth).astype(float)


def posterior_mode(L, flip, seed):
    X = simulate(L, flip, np.random.default_rng(100 * L + seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BoolMF(min_support=1, random_state=seed, **presets.rukat_yau2019).fit(X)
    k = np.asarray(m.n_components_draws_).ravel()
    return int(np.bincount(k).argmax())


def run(flip, repeats=3, ranks=RANKS):
    """Excess of the posterior mode over the true rank, per rank and repeat."""
    return np.array([[posterior_mode(L, flip, s) - L for s in range(repeats)] for L in ranks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    for flip in NOISE:
        excess = run(flip, args.repeats)
        exact = float(np.mean(excess == 0))
        print(f"{flip:.0%} flips: mode - L per rank {list(RANKS)}: "
              f"{[list(map(int, e)) for e in excess]}")
        print(f"  exact in {exact:.0%} of runs, mean excess {excess.mean():+.2f}", flush=True)


if __name__ == "__main__":
    main()
