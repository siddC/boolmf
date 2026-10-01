"""Reproduce Rukat et al. (2017), Section 4.1 (Fig. 4): random matrix factorisation.

An N x N matrix is the Boolean product of random factors of rank L (factor density chosen so the
matrix density is 0.5, or 0.7), each bit is flipped with probability f, and the OrMachine
reconstructs it. The error is the fraction of entries where the rounded reconstruction differs
from the noise-free matrix, averaged over repeats.

Published values were read from the figure (to about +-0.03):
100 x 100, rank 7, density 0.5: 20% flips 0.015, 25% 0.05, 30% 0.15, 35% 0.27, 40% 0.325,
45% 0.405, 50% 0.50. Density 0.7: 0.03, 0.075, 0.165, 0.225, 0.25, 0.29, 0.30.
1000 x 1000, rank 5: about 0 up to 40% flips, about 0.23 at 45%.

The factor prior probability is set from the design density (0.5 or 0.7) rather than from the
noisy matrix: ``presets.rukat2017`` uses ``"empirical"`` (the observed density), but at 45-50%
flips the published errors (0.30 at density 0.7 and 50% flips, that is, predicting every entry
present) are only reached when the prior comes from the noise-free density, so that is what
the paper must have used in this experiment. ``--observed`` uses the observed density instead.

Run: python benchmarks/papers/rukat2017.py [--repeats 10] [--large] [--observed]
"""
import argparse
import warnings

import numpy as np

from boolmf import BayesianBooleanMF, presets

PUBLISHED = {
    (100, 7, 0.5): {0.20: 0.015, 0.25: 0.05, 0.30: 0.15, 0.35: 0.27, 0.40: 0.325, 0.45: 0.405,
                    0.50: 0.50},
    (100, 7, 0.7): {0.20: 0.03, 0.25: 0.075, 0.30: 0.165, 0.35: 0.225, 0.40: 0.25, 0.45: 0.29,
                    0.50: 0.30},
    (1000, 5, 0.5): {0.35: 0.0, 0.40: 0.0, 0.45: 0.23},
}
TOLERANCE = 0.03


def simulate(N, L, density, flip, rng):
    p = np.sqrt(1.0 - (1.0 - density) ** (1.0 / L))
    Z = rng.random((N, L)) < p
    U = rng.random((N, L)) < p
    truth = (Z.astype(int) @ U.T.astype(int)) > 0
    X = np.where(rng.random((N, N)) < flip, ~truth, truth)
    return X.astype(float), truth


def reconstruction_error(model, truth):
    Zbar, Ubar = model.activations_, model.components_
    log_none = np.zeros(truth.shape)
    for k in range(Zbar.shape[1]):
        log_none += np.log1p(-np.clip(np.outer(Zbar[:, k], Ubar[k]), 0, 1 - 1e-12))
    p_or = 1.0 - np.exp(log_none)
    return float(np.mean((p_or >= 0.5) != truth))


def run(N, L, density, flip, repeats, seed=0, observed=False):
    errs = []
    params = dict(presets.rukat2017)
    if not observed:
        p = float(np.sqrt(1.0 - (1.0 - density) ** (1.0 / L)))
        params.update(activation_prior=p, membership_prior=p)
    for r in range(repeats):
        rng = np.random.default_rng(seed + 1000 * r)
        X, truth = simulate(N, L, density, flip, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = BayesianBooleanMF(n_components=L, random_state=seed + r, **params).fit(X)
        errs.append(reconstruction_error(m, truth))
    return float(np.mean(errs)), float(np.std(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--large", action="store_true", help="also run 1000 x 1000, rank 5")
    ap.add_argument("--observed", action="store_true",
                    help="set the factor prior from the observed (noisy) density")
    args = ap.parse_args()
    ok = True
    for (N, L, density), table in PUBLISHED.items():
        if N > 100 and not args.large:
            continue
        for flip, published in table.items():
            mean, sd = run(N, L, density, flip, args.repeats if N <= 100 else 3,
                           observed=args.observed)
            within = abs(mean - published) <= TOLERANCE + 2 * sd / np.sqrt(args.repeats)
            ok &= within
            print(f"N={N} L={L} density={density} flips={flip:.2f}: error {mean:.3f} "
                  f"(sd {sd:.3f}); published {published:.3f}  {'ok' if within else 'OFF'}",
                  flush=True)
    print("all within tolerance" if ok else "some values off")


if __name__ == "__main__":
    main()
