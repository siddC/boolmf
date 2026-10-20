"""Consensus phylons over independent BBMF runs (stability selection across runs).

Each run is a finished ``bbmf_phylons.py`` run on the same P matrix with the same settings
except ``--run-seed`` (the runs may or may not include the refit; only their discovery stage is
used). A phylon is a component of one run's robust set; phylons of different runs are the same
when they are mutual best matches by cell Jaccard (over the genome x gene rectangle of their
majority sets) of at least --match. Linked phylons form a cluster; a cluster found in at least
--min-runs runs is a consensus phylon, with membership and activation probabilities averaged
over its runs.

Outputs (in out_dir)
--------------------
consensus.npz            members (K x genes), activations (genomes x K), n_runs, support (mean
                         discovery support over the runs holding it): pass it to
                         bbmf_phylons.py --pinned for the final refit
consensus_phylons.csv    per consensus phylon: runs holding it, support in each run, smallest
                         pairwise Jaccard between its members, genomes and genes (majority sets)
consensus_summary.json   per run: robust components and the share in the consensus; pairwise
                         reproducibility between runs; consensus phylons by number of runs
log of the final refit command to run next

usage: python bbmf_consensus.py out_dir RUN_DIR RUN_DIR [RUN_DIR ...] [--min-runs 2]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("out", help="output directory")
    p.add_argument("runs", nargs="+", help="finished bbmf_phylons.py run directories")
    p.add_argument("--min-runs", type=int, default=2,
                   help="runs a phylon must be found in to enter the consensus")
    p.add_argument("--match", type=float, default=0.5,
                   help="cell Jaccard at which phylons of two runs are the same")
    a = p.parse_args()
    if len(a.runs) < 2:
        p.error("give at least two runs")
    if not 1 <= a.min_runs <= len(a.runs):
        p.error("--min-runs must be between 1 and the number of runs")
    return a


def load_run(d):
    r = np.load(os.path.join(d, "stage1", "robust.npz"))
    genes = pd.read_csv(os.path.join(d, "genes.csv")).gene.astype(str).tolist()
    genomes = pd.read_csv(os.path.join(d, "stage1_A.csv.gz"), index_col=0, nrows=1) \
        .columns.astype(str).tolist()
    args = json.load(open(os.path.join(d, "diagnostics.json"))).get("args", {})
    return dict(dir=d, members=r["members"].astype(np.float64),
                activations=r["activations"].astype(np.float64), support=r["support"],
                genes=genes, genomes=genomes, args=args)


def majority(run):
    return ((run["activations"] >= 0.5).T.astype(np.float32),
            (run["members"] >= 0.5).astype(np.float32))


def cell_jaccard(Ra, Ga, Rb, Gb):
    inter = (Ra @ Rb.T) * (Ga @ Gb.T)
    sa, sb = Ra.sum(1) * Ga.sum(1), Rb.sum(1) * Gb.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(inter > 0, inter / (sa[:, None] + sb[None, :] - inter), 0.0)


def check_runs(runs):
    ref = runs[0]
    for r in runs[1:]:
        if r["genes"] != ref["genes"] or r["genomes"] != ref["genomes"]:
            raise SystemExit(f"{r['dir']}: genes or genomes differ from {ref['dir']}; the runs "
                             "must use the same P matrix")
        for k in ("zeta", "threshold", "min_freq", "max_freq"):
            if r["args"].get(k) != ref["args"].get(k):
                print(f"warning: {r['dir']} has {k}={r['args'].get(k)} but {ref['dir']} has "
                      f"{ref['args'].get(k)}")
        seeds = [x["args"].get("run_seed", 0) for x in runs]
        if len(set(seeds)) < len(seeds):
            print(f"warning: runs share a --run-seed ({seeds}); they are not independent")


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    runs = [load_run(d) for d in a.runs]
    check_runs(runs)
    sets = [majority(r) for r in runs]
    R = len(runs)
    # nodes: (run, component); edges: mutual best matches between every pair of runs
    offset = np.cumsum([0] + [len(s[0]) for s in sets])
    parent = list(range(int(offset[-1])))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair_J, pairwise = {}, {}
    for i in range(R):
        for j in range(i + 1, R):
            J = cell_jaccard(*sets[i], *sets[j])
            pair_J[(i, j)] = J
            if J.size == 0:
                continue
            bi, bj = J.argmax(1), J.argmax(0)
            ok = [(k, int(bi[k])) for k in range(J.shape[0])
                  if J[k, bi[k]] >= a.match and bj[bi[k]] == k]
            for k, l in ok:
                parent[find(offset[i] + k)] = find(offset[j] + l)
            pairwise[f"{i}->{j}"] = float(np.mean(J.max(1) >= a.match))
            pairwise[f"{j}->{i}"] = float(np.mean(J.max(0) >= a.match))
    clusters = {}
    for r in range(R):
        for k in range(len(sets[r][0])):
            clusters.setdefault(find(offset[r] + k), []).append((r, k))
    cons = []
    for members in clusters.values():
        by_run = {}
        for r, k in members:                     # at most one per run: the best supported
            if r not in by_run or runs[r]["support"][k] > runs[r]["support"][by_run[r]]:
                by_run[r] = k
        if len(by_run) >= a.min_runs:
            cons.append(by_run)
    rows, M, A = [], [], []
    for by_run in cons:
        M.append(np.mean([runs[r]["members"][k] for r, k in by_run.items()], axis=0))
        A.append(np.mean([runs[r]["activations"][:, k] for r, k in by_run.items()], axis=0))
        js = [pair_J[(min(r1, r2), max(r1, r2))][(k1, k2) if r1 < r2 else (k2, k1)]
              for (r1, k1) in by_run.items() for (r2, k2) in by_run.items() if r1 < r2]
        rows.append({"n_runs": len(by_run),
                     "support_mean": float(np.mean([runs[r]["support"][k]
                                                    for r, k in by_run.items()])),
                     **{f"support_run{r}": float(runs[r]["support"][by_run[r]])
                        if r in by_run else np.nan for r in range(R)},
                     **{f"index_run{r}": int(by_run[r]) + 1 if r in by_run else np.nan
                        for r in range(R)},
                     "min_pairwise_jaccard": float(min(js)) if js else np.nan})
    M = np.asarray(M).reshape(len(cons), -1)
    A = np.asarray(A).reshape(len(cons), -1).T
    tab = pd.DataFrame(rows)
    if len(tab):
        order = np.lexsort((-tab["support_mean"].to_numpy(), -tab["n_runs"].to_numpy()))
        tab, M, A = tab.iloc[order].reset_index(drop=True), M[order], A[:, order]
        tab.insert(0, "phylon", [f"consensus_{k + 1}" for k in range(len(tab))])
        tab["n_genomes"] = (A >= 0.5).sum(0)
        tab["n_genes"] = (M >= 0.5).sum(1)
    np.savez_compressed(os.path.join(a.out, "consensus.npz"), members=M, activations=A,
                        n_runs=tab["n_runs"].to_numpy() if len(tab) else np.zeros(0),
                        support=tab["support_mean"].to_numpy() if len(tab) else np.zeros(0))
    tab.to_csv(os.path.join(a.out, "consensus_phylons.csv"), index=False)
    in_cons = [sum(1 for c in cons if r in c) for r in range(R)]
    summary = {"runs": [r["dir"] for r in runs],
               "run_seeds": [r["args"].get("run_seed", 0) for r in runs],
               "robust_per_run": [len(s[0]) for s in sets],
               "share_in_consensus": [in_cons[r] / max(len(sets[r][0]), 1) for r in range(R)],
               "pairwise_reproduced": pairwise, "min_runs": a.min_runs, "match": a.match,
               "consensus_phylons": len(cons),
               "by_n_runs": {int(k): int(v) for k, v in
                             tab["n_runs"].value_counts().sort_index().items()} if len(tab)
               else {}}
    json.dump(summary, open(os.path.join(a.out, "consensus_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    # the refit starts from a run's saved discovery draws; runs from before they were saved
    # (no stage1/draw_states.npz) would start with empty free slots, so prefer the others
    has_states = [os.path.exists(os.path.join(r["dir"], "stage1", "draw_states.npz"))
                  for r in runs]
    best = max(range(R), key=lambda r: (has_states[r], len(sets[r][0])))
    if not has_states[best]:
        print("\nwarning: no run has stage1/draw_states.npz (runs from before it was saved); "
              "the refit will start with empty free slots, and the pinned phylons may absorb "
              "structure the discovery held elsewhere")
    print("\nfinal refit (pins the consensus phylons, starts from the discovery draws of one run;"
          " use that run's settings):\n"
          f"python examples/bbmf_phylons.py P_MATRIX {os.path.join(a.out, 'refit')} "
          f"--discovery {runs[best]['dir']} --pinned {os.path.join(a.out, 'consensus.npz')} "
          f"--zeta {runs[0]['args'].get('zeta')} --holdout {runs[0]['args'].get('holdout')} "
          f"--chains {runs[0]['args'].get('chains')} --run-seed "
          f"{runs[best]['args'].get('run_seed', 0)} --threads N [--compare-L ... --compare-A "
          "...]")


if __name__ == "__main__":
    main()
