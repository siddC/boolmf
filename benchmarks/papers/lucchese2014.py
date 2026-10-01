"""Reproduce PANDA+ (Lucchese, Orlando & Perego 2014, IEEE TKDE 26(12): 2900-2913), Table 4:
the number of patterns k and the normalized cost J_E / J_E(no patterns) found with the MDL
cost J_E on synthetic data.

Generator (Section 4.1, Table 2): N = 10,000 transactions, M = 100 items, K embedded patterns
(5, 10, 15, 20), each with a uniformly drawn number of items in [3, 15] and of transactions in
[500, 5000], chosen uniformly at random; any two patterns share at most half of the smaller
one's items and at most half of its transactions (enforced here by redrawing a pattern that
breaks it); patterns are embedded by OR; each bit is then flipped with probability
n = 0, 3, 5 or 7%. The paper reports one dataset per cell; here each cell is one draw too, so
the ground-truth patterns' own cost on that draw is printed next to PANDA+'s.

Settings (Table 3): PANDA+ is run for up to 64 patterns with the noise thresholds eps_r and
eps_c swept over 0, 0.1, ..., 1 and all three item orders (the paper also tries 20
randomization rounds), and the run with the lowest J_E is kept; PANDA+ with J_E stops by
itself, so k is the number of patterns it returns.

Tolerance: k between K and K + 3 (the published k reaches K + 5), J_E within 0.03 of the
ground truth's J_E on the same draw and within 0.06 of the published value (single draws of the
ground truth's J_E vary by 0.01-0.07).

Results: 14 of 16 cells within tolerance. The two misses are at K = 20: at 3% noise PANDA+
finds 28 patterns with J_E 0.411 (published 23 and 0.41, embedded patterns 0.357), and at 7%
J_E is 0.576 (published 0.60, embedded patterns 0.546). Adding 20 randomized rounds (Section
3.7) to the sweep at 3% gives 25 patterns and 0.398.

Run: python benchmarks/papers/lucchese2014.py [--quick]
"""
import argparse
import time

import numpy as np

from boolmf._algorithms.panda import panda, panda_cost

PUBLISHED = {  # K: [(k, J_E normalized) at n = 0, 3, 5, 7%]
    5: [(5, 0.10), (5, 0.39), (5, 0.51), (5, 0.60)],
    10: [(10, 0.13), (10, 0.36), (11, 0.47), (10, 0.56)],
    15: [(15, 0.16), (16, 0.36), (16, 0.46), (17, 0.55)],
    20: [(20, 0.20), (23, 0.41), (23, 0.50), (25, 0.60)],
}
NOISE = (0.0, 0.03, 0.05, 0.07)
N, M = 10_000, 100


def simulate(K, noise, rng, omega=0.5):
    items, trans = [], []
    while len(items) < K:
        I = rng.choice(M, rng.integers(3, 16), replace=False)
        T = rng.choice(N, rng.integers(500, 5001), replace=False)
        if all(len(np.intersect1d(I, I2)) <= omega * min(len(I), len(I2))
               and len(np.intersect1d(T, T2)) <= omega * min(len(T), len(T2))
               for I2, T2 in zip(items, trans)):
            items.append(I)
            trans.append(T)
    usage = np.zeros((N, K), bool)
    basis = np.zeros((K, M), bool)
    for k in range(K):
        usage[trans[k], k] = True
        basis[k, items[k]] = True
    truth = (usage.astype(np.int64) @ basis.astype(np.int64)) > 0
    X = truth ^ (rng.random(truth.shape) < noise)
    return X, usage, basis


def normalized(X, usage, basis):
    empty = panda_cost(X, np.zeros((X.shape[0], 0), bool), np.zeros((0, X.shape[1]), bool))
    return panda_cost(X, usage, basis) / empty


def best_run(X, eps_grid, orders=("frequency", "couples", "correlation")):
    best = None
    for order in orders:
        for eps_r in eps_grid:
            for eps_c in eps_grid:
                W, H = panda(X, 64, cost="je", eps_r=eps_r, eps_c=eps_c, order=order)
                c = normalized(X, W, H)
                if best is None or c < best[1]:
                    best = (H.shape[0], c, (order, float(eps_r), float(eps_c)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="K = 5 and 10 only")
    args = ap.parse_args()
    eps_grid = np.round(np.arange(0, 1.0001, 0.1), 1)
    ok = True
    for K, row in PUBLISHED.items():
        if args.quick and K > 10:
            continue
        for noise, (k_pub, j_pub) in zip(NOISE, row):
            t = time.time()
            X, usage, basis = simulate(K, noise, np.random.default_rng(1000 * K + int(noise * 100)))
            k, j, setting = best_run(X, eps_grid)
            j_true = normalized(X, usage, basis)
            within = K <= k <= K + 3 and abs(j - j_true) <= 0.03 and abs(j - j_pub) <= 0.06
            ok &= within
            print(f"K={K:2d} n={noise:.0%}: k {k:2d} (published {k_pub}), J_E {j:.3f} (published "
                  f"{j_pub:.2f}, ground truth {j_true:.3f}), {setting} {'ok' if within else 'OFF'} "
                  f"({time.time() - t:.0f} s)", flush=True)
    print("all within tolerance" if ok else "some values off")


if __name__ == "__main__":
    main()
