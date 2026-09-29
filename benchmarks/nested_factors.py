"""v0.1 gate benchmark: recovery on make_nested_factors (8 exclusive groups + 12 overlapping)."""
import argparse
import time

import numpy as np

from boolmf import BoolMF
from boolmf.datasets import make_nested_factors
from boolmf.matching import jaccard_matrix
from boolmf.model_selection import EntryShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-sweeps", type=int, default=4000)
    ap.add_argument("--likelihood", default="or_flip")
    args = ap.parse_args()

    X, truth = make_nested_factors(random_state=args.seed, return_truth=True)
    test = next(EntryShuffleSplit(test_size=0.1, random_state=args.seed).split(X))
    t = time.time()
    m = BoolMF(likelihood=args.likelihood, n_chains=args.chains, n_jobs=args.jobs,
               max_sweeps=args.max_sweeps, n_draws=100, random_state=args.seed).fit(X, mask=test)
    elapsed = time.time() - t

    members, acts = m.binarize_components()
    keep = m.component_flags_ == "robust"
    Jg = jaccard_matrix(truth["activations"].T, acts[:, keep].T)    # carriers
    Jf = jaccard_matrix(truth["members"], members[keep])              # members
    found = [(Jg[k].max() >= 0.8) and (Jf[k, Jg[k].argmax()] >= 0.5) for k in range(len(Jg))]
    ng = truth["n_groups"]
    S = truth["structure"]
    p_true = np.where(S, 0.97, 0.005)
    y = X[test]
    pt = np.clip(p_true[test], 1e-12, 1 - 1e-12)
    true_ll = float(np.mean(np.where(y == 1, np.log(pt), np.log1p(-pt))))
    heldout = m.score(X, entries=test)
    E = m.explained_probability()
    top = E >= 0.95
    calib = float(S[top].mean())

    print(f"seed {args.seed}, {args.likelihood}, {args.chains} chains, {elapsed:.0f}s")
    names, counts = np.unique(m.component_flags_, return_counts=True)
    flags = ", ".join(f"{n} {c}" for n, c in zip(names.tolist(), counts.tolist()))
    print(f"  robust components {m.n_components_}; flags: {flags}")
    print(f"  exclusive groups recovered {sum(found[:ng])}/{ng}; "
          f"overlapping components recovered {sum(found[ng:])}/{len(found) - ng}")
    for name, ok, jg, jf in zip(truth["names"], found, Jg.max(1), Jf.max(1)):
        print(f"    {'yes' if ok else ' no'}  carriers J={jg:.2f} members J={jf:.2f}  {name}")
    print(f"  held-out log predictive density {heldout:.4f} (true model {true_ll:.4f}; "
          f"perplexity gap {abs(-heldout - (-true_ll)):.4f})")
    print(f"  calibration: entries with P(explained) >= 0.95 truly explained: {calib:.4f}")
    print(f"  detection {m.detection_rate_:.4f} background {m.background_rate_:.4f}; "
          f"chain status {list(map(str, m.chain_status_))}")
    print(f"  rhat {({k: round(v, 3) for k, v in m.rhat_.items()})}")
    acc = {k: round(v, 3) for k, v in m.split_merge_acceptance_.items()}
    print(f"  split-merge acceptance {acc}")


if __name__ == "__main__":
    main()
