"""Reproduce Asso (Miettinen, Mielikainen, Gionis, Das & Mannila 2008, The Discrete Basis
Problem, IEEE TKDE 20(10): 1348-1362).

1. Digits, Table 3: reconstruction error d1 = |X - W o H| of Asso at k = 5, 10, 20 on the UCI
   Multiple Features pixel data (2000 x 240, pixel > 0 as 1: 291,654 ones, as in Table 2).
   The paper tuned tau and w+ per dataset without listing the values, so the best error over
   tau = 0.05, 0.10, ..., 1.00 and w+ in {1, 2} (w- = 1) is compared with the published one;
   it must be within 5%.
2. Mushroom, from Belohlavek & Trnecka (2015), Table 4: Asso needs 2, 6 and 36 factors to reach
   coverage c = 1 - E / |X| of 25, 50 and 75% and never reaches 95% (tau = 0.95, w+ = w- = 1,
   60 factors); c counts wrongly covered zeros as errors, which Asso cannot undo.

Run: python benchmarks/papers/miettinen2008.py
"""
import time

import numpy as np
from _datasets import coverage_curve, digits, factors_needed, one_hot

from boolmf import BooleanMF

DIGITS_PUBLISHED = {5: 124_600, 10: 108_500, 20: 87_800}
MUSHROOM_PUBLISHED = [2, 6, 36, None]
TAUS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def digits_errors(ks=(5, 10, 20)):
    X = digits()
    best = {}
    for k in ks:
        for w in (1.0, 2.0):
            for tau in TAUS:
                m = BooleanMF(k, algorithm="asso", threshold=float(tau), positive_weight=w).fit(X)
                if k not in best or m.reconstruction_err_ < best[k][0]:
                    best[k] = (m.reconstruction_err_, float(tau), w)
    return best


def mushroom_counts():
    X = one_hot("mushroom")
    model = BooleanMF(60, algorithm="asso", threshold=0.95)
    W = model.fit_transform(X)
    curve = coverage_curve(X, W.astype(bool), model.components_.astype(bool))
    return factors_needed(curve, (0.25, 0.5, 0.75, 0.95)), float(curve[-1])


def main():
    ok = True
    t = time.time()
    for k, (err, tau, w) in digits_errors().items():
        pub = DIGITS_PUBLISHED[k]
        within = abs(err - pub) <= 0.05 * pub
        ok &= within
        print(f"Digits k={k:2d}: best error {err:,} (tau {tau}, w+ {w:g}), published {pub:,}  "
              f"{'ok' if within else 'OFF'}")
    print(f"  ({time.time() - t:.0f} s)")
    got, final = mushroom_counts()
    within = got == MUSHROOM_PUBLISHED
    ok &= within
    print(f"Mushroom factors for c = 25/50/75/95%: {got} (c = {final:.3f} at 60), published "
          f"{MUSHROOM_PUBLISHED}  {'ok' if within else 'OFF'}")
    print("all within tolerance" if ok else "some values off")


if __name__ == "__main__":
    main()
