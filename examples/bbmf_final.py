"""Final phylons from several refits of the same consensus (the last step of the workflow).

Each refit is a finished ``bbmf_phylons.py --discovery RUN_DIR --pinned consensus.npz`` run:
the same pinned consensus phylons, started from the discovery draws of a different run. The
pinned phylons keep their slot, so phylon k is the same phylon in every refit and its
probabilities can be averaged directly. Averaging over refits removes what depends on the start
(mostly the genes and genomes at a phylon's margin), as the consensus averages over discovery
runs.

A phylon is kept when it is supported (``supported_in_refit``: present in at least half of the
refit's draws) in at least --min-refits refits (default: all of them); its L and A are the mean
over the refits that support it. Tiers come from the consensus: tier 1 is found in every
discovery run of the consensus, tier 2 in fewer (at least --min-runs of bbmf_consensus.py).

Outputs (in out_dir)
--------------------
L.csv.gz, A.csv.gz          kept phylons: genes x phylons and phylons x genomes probabilities,
                            averaged over the refits (minority-coded scale, as in
                            bbmf_phylons.py); phylon names are those of the refits, so phylon_k
                            is row k of consensus_phylons.csv
phylon_gene_content.csv.gz  genes x kept phylons: P(gene present) averaged over each phylon's
                            genomes, averaged over the refits
phylons.csv                 every pinned phylon: kept, tier, consensus runs and support, support
                            in each refit, genes present / absent and genomes of the averaged
                            probabilities (>= 0.5), and agreement between the refits (smallest
                            pairwise Jaccard of the majority sets: cells, genomes, genes)
genes.csv, genomes.csv      phylons per gene and per genome (all kept, tier 1); genomes.csv also
                            has each genome's rates averaged over the refits
summary.json                per refit: start, supported phylons, stuck chains, NMF recovery;
                            kept phylons by tier; agreement between the refits

usage: python bbmf_final.py out_dir REFIT_DIR REFIT_DIR [REFIT_DIR ...] [--min-refits K]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("out", help="output directory")
    p.add_argument("refits", nargs="+",
                   help="finished bbmf_phylons.py refits of the same --pinned consensus")
    p.add_argument("--min-refits", type=int, default=None,
                   help="refits that must support a phylon for it to be kept (default: all)")
    a = p.parse_args()
    if len(a.refits) < 2:
        p.error("give at least two refits")
    if a.min_refits is None:
        a.min_refits = len(a.refits)
    if not 1 <= a.min_refits <= len(a.refits):
        p.error("--min-refits must be between 1 and the number of refits")
    return a


def load_refit(d):
    need = ["phylons.csv", "L.csv.gz", "A.csv.gz", "genes.csv", "genomes.csv",
            "diagnostics.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise SystemExit(f"{d}: missing {', '.join(missing)}; is it a finished refit?")
    diag = json.load(open(os.path.join(d, "diagnostics.json")))
    if not diag.get("stage2"):
        raise SystemExit(f"{d}: no refit stage (run without --skip-refit)")
    L = pd.read_csv(os.path.join(d, "L.csv.gz"), index_col=0)
    A = pd.read_csv(os.path.join(d, "A.csv.gz"), index_col=0)
    content = os.path.join(d, "phylon_gene_content.csv.gz")
    return dict(dir=d, args=diag.get("args", {}), stage2=diag["stage2"],
                phylons=pd.read_csv(os.path.join(d, "phylons.csv")), L=L, A=A,
                genes=pd.read_csv(os.path.join(d, "genes.csv")),
                genomes=pd.read_csv(os.path.join(d, "genomes.csv")),
                content=pd.read_csv(content, index_col=0) if os.path.exists(content) else None)


def check_refits(refits):
    ref = refits[0]
    for r in refits[1:]:
        if r["args"].get("pinned") != ref["args"].get("pinned"):
            raise SystemExit(f"{r['dir']} pins {r['args'].get('pinned')} but {ref['dir']} pins "
                             f"{ref['args'].get('pinned')}; the refits must pin the same "
                             "consensus")
        if not (r["L"].index.equals(ref["L"].index) and r["A"].columns.equals(ref["A"].columns)):
            raise SystemExit(f"{r['dir']}: genes or genomes differ from {ref['dir']}")
        if list(r["L"].columns) != list(ref["L"].columns):
            raise SystemExit(f"{r['dir']}: phylons differ from {ref['dir']}")
        for k in ("zeta", "threshold", "min_freq", "max_freq", "holdout"):
            if r["args"].get(k) != ref["args"].get(k):
                print(f"warning: {r['dir']} has {k}={r['args'].get(k)} but {ref['dir']} has "
                      f"{ref['args'].get(k)}")
    starts = [r["args"].get("discovery") or r["dir"] for r in refits]
    if len(set(starts)) < len(starts):
        print(f"warning: refits share a start ({starts}); averaging them does not average over "
              "starts")


def discovery_runs(pinned):
    """Number of discovery runs behind the consensus (from its summary), or None."""
    path = os.path.join(os.path.dirname(pinned or ""), "consensus_summary.json")
    if pinned and os.path.exists(path):
        return len(json.load(open(path))["runs"])
    return None


def jaccard(a, b):
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def agreement(Ls, As):
    """Smallest pairwise Jaccard between refits of each phylon's majority sets (genomes, genes
    and genome x gene cells), over the given refits."""
    G = [L >= 0.5 for L in Ls]                   # phylons x genes
    R = [A >= 0.5 for A in As]                   # phylons x genomes
    K = Ls[0].shape[0]
    out = {k: np.ones(K) for k in ("cells", "genomes", "genes")}
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            jr, jg = jaccard(R[i], R[j]), jaccard(G[i], G[j])
            ir, ig = (R[i] & R[j]).sum(1), (G[i] & G[j]).sum(1)
            inter = ir * ig
            size = R[i].sum(1) * G[i].sum(1) + R[j].sum(1) * G[j].sum(1) - inter
            jc = np.where(size > 0, inter / np.maximum(size, 1), 1.0)
            out["genomes"] = np.minimum(out["genomes"], jr)
            out["genes"] = np.minimum(out["genes"], jg)
            out["cells"] = np.minimum(out["cells"], jc)
    return out


def gene_lists(L, k, genes, mask, top=50):
    idx = np.flatnonzero((L[k] >= 0.5) & mask)
    idx = idx[np.argsort(-L[k, idx], kind="stable")][:top]
    return ";".join(genes[idx])


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    refits = [load_refit(d) for d in a.refits]
    check_refits(refits)
    R = len(refits)
    names = np.asarray(refits[0]["L"].columns)
    genes = np.asarray(refits[0]["L"].index.astype(str))
    genomes = np.asarray(refits[0]["A"].columns.astype(str))
    absence = (refits[0]["genes"].set_index("gene").reindex(refits[0]["L"].index)["coding"]
               .to_numpy() == "absence")
    Ls = [r["L"].to_numpy().T for r in refits]   # phylons x genes
    As = [r["A"].to_numpy() for r in refits]     # phylons x genomes
    sup = np.column_stack([r["phylons"]["supported_in_refit"].to_numpy(bool) for r in refits])
    support = np.column_stack([r["phylons"]["support"].to_numpy(float) for r in refits])
    n_sup = sup.sum(1)
    keep = n_sup >= a.min_refits
    # average over the refits that support each phylon (all of them by default)
    w = sup.astype(float)
    wsum = np.maximum(w.sum(1), 1)[:, None]
    L = sum(w[:, [i]] * Ls[i] for i in range(R)) / wsum
    A = sum(w[:, [i]] * As[i] for i in range(R)) / wsum
    agree = agreement(Ls, As)
    ptab = refits[0]["phylons"]
    n_runs = ptab["consensus_n_runs"].to_numpy() if "consensus_n_runs" in ptab else None
    runs = discovery_runs(refits[0]["args"].get("pinned"))
    if n_runs is not None:
        full = runs if runs is not None else int(n_runs.max())
        tier = np.where(n_runs >= full, 1, 2)
    else:                                        # pinned the discovery robust set: one tier
        full, tier = None, np.ones(len(names), int)
    pres = ~absence
    tab = pd.DataFrame({"phylon": names, "kept": keep, "tier": np.where(keep, tier, 0)})
    for col in ("consensus_n_runs", "consensus_support"):
        if col in ptab:
            tab[col] = ptab[col].to_numpy()
    tab["refits_supporting"] = n_sup
    for i in range(R):
        tab[f"support_refit{i}"] = support[:, i]
    tab["support_mean"] = support.mean(1)
    tab["n_genes_present"] = ((L >= 0.5) & pres).sum(1)
    tab["n_genes_absent"] = ((L >= 0.5) & absence).sum(1)
    tab["n_genomes"] = (A >= 0.5).sum(1)
    tab["expected_genes"] = L.sum(1).round(2)
    tab["expected_genomes"] = A.sum(1).round(2)
    tab["min_cell_jaccard_refits"] = agree["cells"].round(4)
    tab["min_genome_jaccard_refits"] = agree["genomes"].round(4)
    tab["min_gene_jaccard_refits"] = agree["genes"].round(4)
    tab["genes_present_top50"] = [gene_lists(L, k, genes, pres) for k in range(len(names))]
    tab["genes_absent_top50"] = [gene_lists(L, k, genes, absence) for k in range(len(names))]
    tab.to_csv(os.path.join(a.out, "phylons.csv"), index=False)
    kn = names[keep]
    pd.DataFrame(L[keep].T, index=refits[0]["L"].index, columns=kn).round(4).to_csv(
        os.path.join(a.out, "L.csv.gz"), index_label="gene")
    pd.DataFrame(A[keep], index=kn, columns=refits[0]["A"].columns).round(4).to_csv(
        os.path.join(a.out, "A.csv.gz"), index_label="phylon")
    if all(r["content"] is not None for r in refits):
        C = sum(w[:, [i]] * refits[i]["content"].to_numpy().T for i in range(R)) / wsum
        pd.DataFrame(C[keep].T, index=refits[0]["content"].index, columns=kn).round(4).to_csv(
            os.path.join(a.out, "phylon_gene_content.csv.gz"), index_label="gene")
    t1 = keep & (tier == 1)
    g = refits[0]["genes"][["gene", "frequency", "coding"]].copy()
    g["n_phylons"] = (L[keep] >= 0.5).sum(0)
    g["n_tier1_phylons"] = (L[t1] >= 0.5).sum(0)
    g.to_csv(os.path.join(a.out, "genes.csv"), index=False)
    s = pd.DataFrame({"genome": genomes, "n_phylons": (A[keep] >= 0.5).sum(0),
                      "n_tier1_phylons": (A[t1] >= 0.5).sum(0)})
    rate_cols = [c for c in refits[0]["genomes"].columns
                 if c.startswith(("detection_", "background_"))]
    for c in rate_cols:
        s[c] = np.mean([r["genomes"].set_index("genome").reindex(genomes)[c].to_numpy()
                        for r in refits], axis=0)
    s.to_csv(os.path.join(a.out, "genomes.csv"), index=False)
    def within(x, y, tol=0.1):                   # kept phylons whose sizes differ <= tol
        x, y = x[keep], y[keep]
        return float(np.mean(np.abs(x - y) / np.maximum(np.maximum(x, y), 1) <= tol)) \
            if keep.any() else None

    pair_size = [{"refits": f"{i}-{j}",
                  "genes_within_10pct": within((Ls[i] >= 0.5).sum(1), (Ls[j] >= 0.5).sum(1)),
                  "genomes_within_10pct": within((As[i] >= 0.5).sum(1), (As[j] >= 0.5).sum(1))}
                 for i in range(R) for j in range(i + 1, R)]
    summary = {
        "refits": [{"dir": r["dir"], "start": r["args"].get("discovery"),
                    "run_seed": r["args"].get("run_seed"),
                    "supported": int(sup[:, i].sum()),
                    "stuck_chains": [c for c, x in enumerate(r["stage2"].get("chains", []))
                                     if x != "ok"],
                    "free_slot_robust_unmatched": r["stage2"].get(
                        "robust_in_free_slots_unmatched"),
                    "compare_recovered": {k: (r["stage2"].get("compare_match") or {}).get(k)
                                          for k in ("recovered_0.9", "recovered_0.5")}}
                   for i, r in enumerate(refits)],
        "pinned": refits[0]["args"].get("pinned"), "discovery_runs": full,
        "min_refits": a.min_refits, "phylons_pinned": int(len(names)),
        "kept": int(keep.sum()),
        "kept_by_tier": {int(t): int(((tier == t) & keep).sum()) for t in np.unique(tier)},
        "dropped_by_tier": {int(t): int(((tier == t) & ~keep).sum()) for t in np.unique(tier)},
        "supported_by_n_refits": {int(k): int((n_sup == k).sum()) for k in range(R + 1)},
        "agreement_kept": {
            "median_min_cell_jaccard": float(np.median(agree["cells"][keep])) if keep.any()
            else None,
            "share_min_cell_jaccard_ge_0.8": float(np.mean(agree["cells"][keep] >= 0.8))
            if keep.any() else None,
            "pairs": pair_size},
        "genomes_with_a_phylon": float(np.mean((A[keep] >= 0.5).any(0))) if keep.any() else 0.0,
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
    sys.exit(0)
