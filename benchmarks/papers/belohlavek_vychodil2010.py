"""Reproduce the GreConD factor counts (Belohlavek & Vychodil 2010, Algorithm 2) published in
Belohlavek & Trnecka (2015), From-below Boolean matrix factorization algorithm based on MDL,
J. Comput. Syst. Sci. 81: 1678-1697, Table 4: the number of factors GreConD needs to cover 25,
50, 75, 95 and 100% of the ones.

Data: UCI Mushroom (8124 x 119), Tic-tac-toe (958 x 29) and Chess (3196 x 75), every attribute
value (class included) as a binary column. The paper's Chess matrix has 76 columns, so its
partial counts can differ by one; the exact cover can differ by one with column order, which
decides ties.

Run: python benchmarks/papers/belohlavek_vychodil2010.py
"""
import time

from _datasets import coverage_curve, factors_needed, one_hot

from boolmf import BooleanMF

PUBLISHED = {"mushroom": [3, 7, 24, 63, 120], "tic_tac_toe": [5, 12, 19, 28, 32],
             "chess": [1, 4, 15, 46, 124]}
TOLERANCE = {"mushroom": [0, 0, 0, 1, 1], "tic_tac_toe": [0, 0, 0, 0, 0],
             "chess": [1, 1, 1, 1, 1]}


def run(name):
    X = one_hot(name)
    model = BooleanMF(algorithm="grecond")
    W = model.fit_transform(X)
    return factors_needed(coverage_curve(X, W.astype(bool), model.components_.astype(bool)))


def main():
    ok = True
    for name, published in PUBLISHED.items():
        t = time.time()
        got = run(name)
        within = all(abs(g - p) <= tol for g, p, tol in zip(got, published, TOLERANCE[name]))
        ok &= within
        print(f"{name:12s} factors for 25/50/75/95/100%: {got}, published {published}  "
              f"{'ok' if within else 'OFF'} ({time.time() - t:.1f} s)")
    print("all within tolerance" if ok else "some values off")


if __name__ == "__main__":
    main()
