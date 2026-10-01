"""Reproduce Wagala, Samur & Parmigiani (2026), A Bayesian Boolean Matrix Factorization with
Application to Copy Number Analysis in Cancer (arXiv:2606.17491), Table 1.

Two simulated copy-number matrices with R = 4 factors (``data/wagala2026_*.csv``):

* Scenario 1 (70 patients x 44 chromosome arms, 20% of entries flipped), regenerated with the
  authors' simulation script (``scripts/simulation-study-I.R`` in their BBMF package, seeds as
  given there); it equals, entry for entry, the matrices shown in the paper's Figure 2.
* Scenario 2 (62 x 44), read from the paper's Figure 3 (the factors W and H and the noisy
  matrix; 10% of entries differ from W o H). This is not the matrix behind Table 1: Asso, and
  the authors' R implementations of Asso and GreConD+, give different numbers on it (below).

BBMF (``presets.wagala2026``, 4 chains of 100,000 sweeps, the first 20,000 discarded, all
started from Asso) is scored by its MAP draw: the reconstruction W o H of the draw with the
highest log posterior is compared with the noise-free matrix (specificity, F1, MCC and error
rate). Asso is ``BooleanMF(4, algorithm="asso")`` with the paper's settings (threshold 0.5,
unit weights). GreConD+ is not implemented in boolmf.

Published (Table 1)                 Specificity  F1     MCC    Error rate
  Scenario 1  Asso                  0.910        0.830  0.767  0.095
              BBMF                  0.960        0.928  0.903  0.039
  Scenario 2  Asso                  0.968        0.910  0.887  0.037
              BBMF                  0.984        0.952  0.940  0.019

The MAP draw is a single draw, so one run is noisy (on Scenario 1 the error rate ranged from
0.038 to 0.050 over three seeds); BBMF is run with several seeds and the mean is compared.
Tolerance (Scenario 1): Asso exact (to the 3 published decimals), the mean of BBMF over the
seeds within 0.01 per metric. Scenario 2 is reported without a pass/fail check, since its data
differ.

Run: python benchmarks/papers/wagala2026.py [--sweeps 100000] [--jobs 4] [--seeds 3]
[--scenario 1]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from boolmf import BayesianBooleanMF, BooleanMF, presets

DATA = Path(__file__).resolve().parent / "data"
PUBLISHED = {
    (1, "asso"): (0.910, 0.830, 0.767, 0.095), (1, "bbmf"): (0.960, 0.928, 0.903, 0.039),
    (2, "asso"): (0.968, 0.910, 0.887, 0.037), (2, "bbmf"): (0.984, 0.952, 0.940, 0.019),
}


def load(scenario):
    def read(name):
        return np.loadtxt(DATA / f"wagala2026_scenario{scenario}_{name}.csv", delimiter=",",
                          dtype=int)
    return read("xsim"), read("xtruth").astype(bool)


def metrics(Z, truth):
    """Specificity, F1, MCC and error rate of a reconstruction against the truth."""
    Z = np.asarray(Z, bool)
    tp, fp = np.sum(Z & truth), np.sum(Z & ~truth)
    fn, tn = np.sum(~Z & truth), np.sum(~Z & ~truth)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tn / (tn + fp), 2 * precision * recall / (precision + recall), mcc,
            (fp + fn) / truth.size)


def fit_bbmf(X, sweeps=100_000, jobs=4, seed=0):
    burn = sweeps // 5
    params = {**presets.wagala2026, "burn_in": burn, "n_draws": sweeps - burn,
              "max_sweeps": sweeps}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BayesianBooleanMF(n_components=4, n_jobs=jobs, random_state=seed,
                                 **params).fit(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenario", type=int, choices=(1, 2), default=None)
    args = ap.parse_args()
    ok = True
    for scenario in (args.scenario,) if args.scenario else (1, 2):
        X, truth = load(scenario)
        asso = BooleanMF(4, algorithm="asso", threshold=0.5)
        got = metrics(asso.inverse_transform(asso.fit_transform(X)), truth)
        pub = PUBLISHED[(scenario, "asso")]
        within = scenario == 2 or max(abs(g - p) for g, p in zip(got, pub)) <= 0.0005
        ok &= within
        print(f"Scenario {scenario} Asso: {_fmt(got)}   published {_fmt(pub)}"
              + ("" if scenario == 2 else ("  ok" if within else "  OFF")), flush=True)
        runs = []
        for seed in range(args.seeds):
            t = time.time()
            m = fit_bbmf(X, args.sweeps, args.jobs, seed)
            runs.append(metrics(m.map_activations_.astype(int) @ m.map_components_ > 0, truth))
            print(f"  BBMF seed {seed}: {_fmt(runs[-1])}  ({time.time() - t:.0f} s)", flush=True)
        mean = np.mean(runs, axis=0)
        pub = PUBLISHED[(scenario, "bbmf")]
        within = scenario == 2 or max(abs(g - p) for g, p in zip(mean, pub)) <= 0.01
        ok &= within
        print(f"Scenario {scenario} BBMF mean: {_fmt(mean)}   published {_fmt(pub)}"
              + ("" if scenario == 2 else ("  ok" if within else "  OFF")), flush=True)
    print("Scenario 1 within tolerance" if ok else "some values off")


def _fmt(values):
    return " ".join(f"{v:.3f}" for v in values)


if __name__ == "__main__":
    main()
