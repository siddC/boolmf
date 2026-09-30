"""Reproduce Wood, Griffiths & Ghahramani (2006), Section 5, second experiment (Fig. 4).

Four causal structures over 6 observed variables (degree 1, K = 6; disconnected, K = 4;
undercomplete, K = 4; overcomplete, K = 8) generate 10 datasets each: T = 150 trials, each hidden
cause active with probability p = 0.1, noisy-OR with lambda = 0.9 and epsilon = 0.01. The Gibbs
sampler for the infinite model (alpha = 3, the other parameters fixed at their true values)
starts from an empty Z. For a run of n iterations the expected ZZ^T is the average over its n
samples, and

* in-degree error = sum |diag(ZZ^T) - diag(E[ZZ^T])|,
* structure error = sum over the upper triangle of |ZZ^T - E[ZZ^T]|.

The structures were read from the figure's graphs (by tracing the edges in the rendered page).
Published values (Gibbs, mean and standard deviation over the 10 datasets) were read from the
figure's error bars, to about +-0.1 for the first two structures and +-0.5 for the others. The
paper's first experiment (Fig. 3) is not reproduced: it does not state the number of observed
variables.

A run is compared with the paper one-sidedly (the paper's claim is that the Gibbs sampler
recovers structures close to the truth, so a lower error is not a failure to reproduce): our
mean over the datasets must not exceed the published mean by more than the reading error plus
twice the standard error of the difference of two means over 10 datasets each.

With ``births="metropolis"`` the same runs use the Metropolis-Hastings births of Meeds et al.
(2007): new causes proposed with their trial activations drawn from the prior, which here
(150 trials, p = 0.1) are almost never accepted, so causes are found slowly.

Run: python benchmarks/papers/wood2006.py [--datasets 10] [--births enumerate] [--seed 0]
"""
import argparse
import time
import warnings

import numpy as np

from boolmf import BayesianBooleanMF, presets

STRUCTURES = {
    "degree1": np.eye(6, dtype=int),
    "disconnected": np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1],
                              [0, 0, 1, 0], [0, 0, 0, 1]]),
    "undercomplete": np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0],
                               [1, 1, 1, 0], [0, 0, 1, 1]]),
    "overcomplete": np.array([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 1, 1, 0],
                              [1, 0, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 1],
                              [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]),
}
# (in-degree mean, sd, structure mean, sd) after n iterations, Gibbs sampler, Fig. 4
PUBLISHED = {
    "degree1": {1: (0.1, 0.25, 0.1, 0.3), 10: (0.0, 0.05, 0.0, 0.05),
                100: (0.15, 0.25, 0.05, 0.2), 1000: (0.35, 0.2, 0.3, 0.45)},
    "disconnected": {1: (1.2, 1.4, 1.6, 1.6), 10: (0.45, 0.4, 0.3, 0.4),
                     100: (0.85, 1.6, 0.55, 1.5), 1000: (0.05, 0.1, 0.05, 0.1)},
    "undercomplete": {1: (3.2, 1.4, 5.9, 1.2), 10: (3.05, 0.6, 5.5, 1.8),
                      100: (3.0, 0.3, 5.7, 2.0), 1000: (3.0, 0.8, 5.4, 2.2)},
    "overcomplete": {1: (5.2, 1.5, 9.5, 3.0), 10: (4.3, 1.6, 8.4, 2.5),
                     100: (3.6, 1.5, 6.2, 2.5), 1000: (2.8, 1.0, 3.8, 2.0)},
}
ITERATIONS = (1, 10, 100, 1000)
T, P, LAMBDA, EPSILON, ALPHA = 150, 0.1, 0.9, 0.01, 3.0


def simulate(Zw, rng):
    """Trials x observed variables."""
    Y = rng.random((Zw.shape[1], T)) < P
    p1 = 1.0 - (1.0 - EPSILON) * (1.0 - LAMBDA) ** (Zw @ Y)
    return (rng.random(p1.shape) < p1).astype(float).T


def errors(model, Zw, iterations):
    """In-degree and structure errors of the first n samples, for each n in iterations."""
    n_obs = Zw.shape[0]
    truth = Zw @ Zw.T
    iu = np.triu_indices(n_obs, 1)
    acc = np.zeros((n_obs, n_obs))
    out = {}
    for s, d in enumerate(model._draws_, start=1):
        U = np.unpackbits(d["U"], axis=0, count=n_obs).astype(float)   # observed x causes
        acc += U @ U.T
        if s in iterations:
            E = acc / s
            out[s] = (float(np.abs(np.diag(truth) - np.diag(E)).sum()),
                      float(np.abs(truth[iu] - E[iu]).sum()))
    return out


def run(name, n_datasets=10, births="enumerate", iterations=ITERATIONS, seed=0):
    """Mean and sd over datasets of (in-degree error, structure error) per iteration count."""
    Zw = STRUCTURES[name]
    params = {**presets.wood2006, "births": births, "detection_prior": LAMBDA,
              "birth_members": "gibbs" if births == "enumerate" else "exact",
              "background_prior": EPSILON, "activation_prior": P, "alpha_prior": ALPHA,
              "burn_in": 0, "n_draws": max(iterations), "thin": 1,
              "max_sweeps": max(iterations) + 1}
    res = {n: [] for n in iterations}
    for r in range(n_datasets):
        X = simulate(Zw, np.random.default_rng(seed + 1000 * r))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = BayesianBooleanMF(random_state=seed + r, **params).fit(X)
        for n, e in errors(m, Zw, iterations).items():
            res[n].append(e)
    return {n: (*np.mean(v, 0), *np.std(v, 0)) for n, v in res.items()}


def slack(name, sd_published, sd_ours, n_ours):
    reading = 0.1 if name in ("degree1", "disconnected") else 0.5
    return reading + 2 * np.sqrt(sd_published ** 2 / 10 + sd_ours ** 2 / n_ours)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=10)
    ap.add_argument("--births", default="enumerate", choices=["enumerate", "metropolis"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ok = True
    for name in STRUCTURES:
        t0 = time.time()
        res = run(name, args.datasets, args.births, seed=args.seed)
        for n in ITERATIONS:
            ind, struct, ind_sd, struct_sd = res[n]
            pi, pi_sd, ps, ps_sd = PUBLISHED[name][n]
            within = (ind <= pi + slack(name, pi_sd, ind_sd, args.datasets)
                      and struct <= ps + slack(name, ps_sd, struct_sd, args.datasets))
            ok &= within
            print(f"{name:13s} {n:5d} iterations: in-degree {ind:5.2f} ({ind_sd:.2f}), published "
                  f"{pi:.2f} ({pi_sd:.2f}); structure {struct:5.2f} ({struct_sd:.2f}), published "
                  f"{ps:.2f} ({ps_sd:.2f})  {'ok' if within else 'OFF'}", flush=True)
        print(f"  ({time.time() - t0:.0f} s)")
    print("all within tolerance" if ok else "some values off")


if __name__ == "__main__":
    main()
