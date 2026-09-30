"""MEBF (Wan et al. 2020, AAAI; arXiv:1909.03991), Fig. 4 and Table 1: not reproduced.

``BooleanMF(algorithm="mebf")`` implements the paper's Algorithms 1-3. The published results
come from the authors' R code (github.com/clwan/MEBF), which differs from the paper in its
seeds, expansion threshold, selection score, weak-signal rule and stopping rule, so neither
table nor figure is expected to match; this script records how far apart they are.

Simulation (Fig. 4): X = U o V with U (n x 5) and V (5 x n) Bernoulli(p0), p0 = 0.2 ("low
density") or 0.4 ("high density"), each bit flipped with probability p = 0 or 0.01;
n = 100 (50 runs) and 1000 (10 runs here). Error after k patterns:
|U o V XOR A o B| / |U o V|. The paper does not give t; 0.6 is the value it uses for its
single-cell data.

Published MEBF errors, read from Fig. 4 (+-0.03) at k = 1, 2, 3, ...:
  100,  low,  no noise: 0.68 0.41 0.18 0.00
  100,  low,  noise:    0.70 0.50 0.41 0.38 0.33 0.16 0.15
  100,  high, no noise: 0.57 0.27 0.12 0.01
  1000, low,  no noise: 0.66 0.42 0.17 0.00
  1000, high, no noise: 0.58 0.29 0.12 0.02

Results (t = 0.6), k = 1..5:
  100,  low,  no noise: 0.73 0.50 0.32 0.16 0.05
  100,  low,  noise:    0.76 0.57 0.42 0.34 0.32
  100,  high, no noise: 0.46 0.35 0.30 0.27 0.24
  1000, low,  no noise: 0.76 0.55 0.36 0.18 0.03
  1000, low,  noise:    0.78 0.59 0.52 0.52 0.52
  1000, high, no noise: 0.48 0.36 0.32 0.29 0.27

Without noise at low density the paper's algorithm follows the published curve about one
pattern behind (five random patterns need five components, so the published error of 0 at
four patterns cannot come from this generator). At high density the first pattern already
has density 0.7 against 0.4 in the figure, so the published data were probably generated
differently. With noise, once the large patterns are removed the median column or row of the
residual is mostly noise, and MEBF as written adds tiny patterns that lower the error by a few
entries each, so weak signal detection, which would find the remaining true patterns, is never
reached (1000 x 1000: the error stays at 0.52).

Table 1 (Chicago crime, head and neck single-cell data): the crime matrix is not public in the
form used; on the single-cell matrix shipped with the authors' code the published density
(2.06e-4, about 6 ones in 6246 x 5 entries) is impossible under the paper's own definition, and
the authors' code gives coverage 0.27 and 0.39 at k = 5 and 20 against 0.496 and 0.626.

Run: python benchmarks/papers/wan2020.py
"""
import numpy as np

from boolmf import BooleanMF

SETTINGS = [(100, 0.2, 0.0), (100, 0.2, 0.01), (100, 0.4, 0.0), (100, 0.4, 0.01),
            (1000, 0.2, 0.0), (1000, 0.2, 0.01), (1000, 0.4, 0.0), (1000, 0.4, 0.01)]


def simulate(n, p0, p, rng, k=5):
    U = rng.random((n, k)) < p0
    V = rng.random((k, n)) < p0
    truth = (U.astype(int) @ V.astype(int)) > 0
    return truth ^ (rng.random(truth.shape) < p), truth


def errors(n, p0, p, t=0.6, k=5, runs=None):
    """Mean error after 1..k patterns (a run that stops early keeps its last error)."""
    runs = runs or (50 if n <= 100 else 10)
    out = np.zeros((runs, k))
    for r in range(runs):
        X, truth = simulate(n, p0, p, np.random.default_rng(r))
        m = BooleanMF(k, algorithm="mebf", threshold=t)
        W = m.fit_transform(X).astype(bool)
        H = m.components_.astype(bool)
        covered = np.zeros(X.shape, bool)
        err = []
        for j in range(H.shape[0]):
            covered |= np.outer(W[:, j], H[j])
            err.append(np.count_nonzero(covered ^ truth) / truth.sum())
        err += [err[-1] if err else 1.0] * (k - len(err))
        out[r] = err
    return out.mean(0)


def main():
    for n, p0, p in SETTINGS:
        e = errors(n, p0, p)
        print(f"n={n:4d} {'low ' if p0 < 0.3 else 'high'} density, noise {p:.2f}: "
              + " ".join(f"{v:.2f}" for v in e), flush=True)


if __name__ == "__main__":
    main()
