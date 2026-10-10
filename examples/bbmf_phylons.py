"""Phylons from a gene presence/absence matrix with BayesianBooleanMF, in two stages.

Input: a P matrix, genes x genomes (rows: genes, columns: genomes; 1 = gene present), as
.csv/.csv.gz/.tsv (first column = gene names), .parquet, .feather or a pickled DataFrame.

What it does
------------
1. Prepares the matrix: binarizes it (> 0 is present, NaN is missing), keeps genes with
   frequency within [--min-freq, --max-freq], holds out a random --holdout share of the cells
   for evaluation, and minority-codes it: genes present in less than --threshold of the
   genomes (training cells) are presence-coded (1 = present), the others absence-coded
   (1 = absent). The two codings get their own detection and background rates per genome
   (``feature_groups``).
2. Discovery: BayesianBooleanMF on the minority-coded matrix (genomes x genes), coarsened
   posterior (``likelihood_power`` --zeta), per-genome rates per coding, Indian buffet process
   over at most --max-components slots, --chains chains started from binarized NMF fits
   (rank --init-rank) with population moves every --population-every sweeps, --sweeps sweeps
   (burn-in half). Robust components: present in at least half of all stored draws and in at
   least two chains, with at least 3 genomes and 3 genes (the limit of detection).
3. Refit: the robust components pinned (``pin_init=True``), each chain starting from its own
   draw of their probabilities, plus --free-slots free slots for the remaining structure;
   independent chains (no population moves), --refit-sweeps sweeps. The pinned components
   keep their slot in every chain and draw, so the split-chain R-hat of every entry of L and
   A is a fair convergence check (over the draws in which the component is present). These
   are the phylons; a pinned component that the refit empties in most draws
   (``supported_in_refit`` False in phylons.csv) is not supported once the others are fixed.

In phylon notation P = L o A (Boolean product: P[g, s] = OR_k L[g, k] AND A[k, s]), with P the
minority-coded matrix (genes x genomes), L genes x phylons and A phylons x genomes. boolmf
writes X = Z o U^T with X genomes x genes, so Z = A^T and U = L. L and A here are posterior
probabilities; the exact posterior predictive is the average over draws (predictive.npy), and
OR-ing L and A reproduces it only approximately (it treats entries as independent).

Outputs (in out_dir)
--------------------
L.csv.gz, A.csv.gz          refit phylons: genes x phylons and phylons x genomes probabilities
stage1_L.csv.gz, stage1_A.csv.gz   the robust components of the discovery stage
phylons.csv                 per phylon: support, genes present / absent, genomes, R-hat
genes.csv, genomes.csv      coding and frequency of every gene; rates of every genome
predictive.npy              posterior predictive P(gene present), genes x genomes, float32,
                            rows and columns in the order of genes.csv and genomes.csv
heldout.npy                 the held-out cells (genes x genomes, bool)
diagnostics.json            convergence, held-out log-likelihood, precision and recall
log.txt                     progress
checkpoints/                resumable state; rerunning the same command resumes

usage: python bbmf_phylons.py P.csv out_dir [--threads 32] [options]   (--help for all)
"""

import argparse
import json
import os
import sys
import threading
import time


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("P", help="P matrix file (genes x genomes)")
    p.add_argument("out", help="output directory")
    p.add_argument("--genomes-as-rows", action="store_true",
                   help="the file has genomes as rows and genes as columns")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="cores to use")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="genes at or above this frequency are absence-coded")
    p.add_argument("--min-freq", type=float, default=0.01)
    p.add_argument("--max-freq", type=float, default=0.99)
    p.add_argument("--holdout", type=float, default=0.02,
                   help="share of cells held out for evaluation (0: none)")
    p.add_argument("--seed", type=int, default=0, help="seed of the held-out cells")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--sweeps", type=int, default=6000, help="discovery sweeps per chain")
    p.add_argument("--refit-sweeps", type=int, default=3000, help="refit sweeps per chain")
    p.add_argument("--max-components", type=int, default=2000,
                   help="component slots in the discovery stage")
    p.add_argument("--free-slots", default="auto",
                   help="free slots next to the pinned phylons in the refit; 'auto': twice "
                        "the mean number of non-robust components per discovery draw, at "
                        "least 100")
    p.add_argument("--init-rank", type=int, default=25, help="rank of the NMF starts")
    p.add_argument("--zeta", type=float, default=0.8, help="likelihood power (coarsening)")
    p.add_argument("--population-every", type=int, default=10,
                   help="sweeps between population moves in the discovery stage (0: off)")
    p.add_argument("--store-draws", type=int, default=100, help="stored draws per chain")
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--quick", action="store_true",
                   help="a short run to check the pipeline (200 + 100 sweeps)")
    a = p.parse_args()
    if a.quick:
        a.sweeps, a.refit_sweeps, a.store_draws = 200, 100, 20
    return a


ARGS = parse()
# thread limits must be set before numpy, numba and BLAS start
for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = str(max(1, ARGS.threads))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boolmf import BayesianBooleanMF  # noqa: E402
from boolmf.consensus import RobustComponents, _cell_jaccard  # noqa: E402

BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
MIN_SIZE = 3
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "a")
T0 = time.time()


def log(*msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - T0) / 60:7.1f} min] " + \
        " ".join(str(m) for m in msg)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


class Watcher(threading.Thread):
    """Logs every checkpoint written under a directory, with an estimate of the time left."""

    def __init__(self, path, stage, total_sweeps, every):
        super().__init__(daemon=True)
        self.path, self.stage, self.total, self.every = path, stage, total_sweeps, every
        self.stop = threading.Event()

    def run(self):
        seen, start, n = {}, time.time(), 0
        while not self.stop.wait(30):
            if not os.path.isdir(self.path):
                continue
            for f in os.listdir(self.path):
                if not f.endswith(".pkl"):
                    continue
                t = os.path.getmtime(os.path.join(self.path, f))
                if seen.get(f) != t:
                    seen[f] = t
                    n += 1
            if n and n != getattr(self, "_logged", 0):
                self._logged = n
                files = max(1, len(seen))
                sweep = min(self.total, n // files * self.every)
                rate = (time.time() - start) / max(sweep, 1)
                left = (self.total - sweep) * rate / 3600
                log(f"{self.stage}: checkpoint at about sweep {sweep}/{self.total}; "
                    f"about {left:.1f} h left at this pace")


def watch(path, stage, total):
    w = Watcher(path, stage, total, ARGS.checkpoint_every)
    w.start()
    return w


# ----------------------------------------------------------------------------- data
def read_matrix(path):
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    elif low.endswith(".feather"):
        df = pd.read_feather(path)
    elif low.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(path)
    elif low.endswith((".tsv", ".tsv.gz", ".txt")):
        df = pd.read_csv(path, sep="\t", index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    if isinstance(df.index, pd.RangeIndex) and df.shape[1] and df.iloc[:, 0].dtype == object:
        df = df.set_index(df.columns[0])         # names stored as the first column
    if ARGS.genomes_as_rows:
        df = df.T
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def prepare():
    df = read_matrix(ARGS.P)
    vals = df.to_numpy(dtype=float)
    missing = np.isnan(vals)
    X = np.where(missing, 0, vals > 0).astype(bool).T            # genomes x genes
    missing = missing.T
    genes, genomes = df.index.to_numpy(), df.columns.to_numpy()
    obs = ~missing
    freq_all = (X & obs).sum(0) / np.maximum(obs.sum(0), 1)
    keep = (freq_all >= ARGS.min_freq) & (freq_all <= ARGS.max_freq)
    log(f"P: {len(genes)} genes x {len(genomes)} genomes; {int((~keep).sum())} genes outside "
        f"[{ARGS.min_freq}, {ARGS.max_freq}] dropped; {int(missing.sum())} missing cells")
    X, missing, genes = X[:, keep], missing[:, keep], genes[keep]
    rng = np.random.default_rng(ARGS.seed)
    test = (rng.random(X.shape) < ARGS.holdout) & ~missing
    train = ~test & ~missing
    freq = (X & train).sum(0) / np.maximum(train.sum(0), 1)
    absence = freq >= ARGS.threshold
    Xm = np.where(absence[None, :], ~X, X)                       # minority-coded
    log(f"kept {len(genes)} genes ({int((~absence).sum())} presence-coded, "
        f"{int(absence.sum())} absence-coded at threshold {ARGS.threshold}); "
        f"{int(test.sum())} held-out cells; minority-coded density "
        f"{Xm[train].mean():.3f}")
    return dict(X=X, Xm=Xm, test=test, mask=test | missing, genes=genes, genomes=genomes,
                freq=freq, absence=absence)


# ----------------------------------------------------------------------------- evaluation
def per_cell_rates(m, coding):
    """Per-genome rates of each cell's coding, shape (genomes, genes)."""
    labels = list(m.feature_groups_)
    g = np.array([labels.index(c) for c in coding])
    return m.detection_rate_per_sample_[:, g], m.background_rate_per_sample_[:, g]


def heldout_ll(P1, D):
    """Mean log-likelihood per held-out cell, overall and by gene-frequency band."""
    if not D["test"].any():
        return None
    P1 = np.clip(P1, 1e-12, 1 - 1e-12)
    ll = np.where(D["Xm"], np.log(P1), np.log1p(-P1))
    out = {"all": float(ll[D["test"]].mean())}
    for lo, hi in BANDS:
        cols = (D["freq"] >= lo) & (D["freq"] < hi)
        sel = D["test"] & cols[None, :]
        out[f"{lo:.0%}-{min(hi, 1):.0%}"] = float(ll[sel].mean()) if sel.any() else None
    return out


def precision_recall(rc, D):
    """Cells predicted present by the majority sets (covered on presence-coded genes, not
    covered on absence-coded genes) against the held-out cells of the original matrix."""
    if not D["test"].any() or rc.n_components == 0:
        return None
    M, Aa = rc.majority()
    covered = (Aa.astype(np.float32) @ M.astype(np.float32)) > 0
    pred = covered ^ D["absence"][None, :]
    t, y = D["test"], D["X"]
    tp = int((pred & y)[t].sum())
    fp = int((pred & ~y)[t].sum())
    fn = int((~pred & y)[t].sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12)}


def core_margin(rc):
    """R-hat below 1.1 for core entries (probability >= 0.9) and margin entries (0.1 to 0.9)."""
    out = {}
    for name, P, R in (("members", rc.members, rc.member_rhat),
                       ("activations", rc.activations, rc.activation_rhat)):
        for part, sel in (("core", P >= 0.9), ("margin", (P > 0.1) & (P < 0.9))):
            r = R[sel]
            r = r[~np.isnan(r)]
            out[f"{name}_{part}_entries"] = int(sel.sum())
            out[f"{name}_{part}_rhat_below_1.1"] = float(np.mean(r < 1.1)) if r.size else None
    return out


def chain_spread(m):
    tr = m.log_likelihood_trace_
    half = tr[:, tr.shape[1] // 2:]
    means = np.nanmean(half, axis=1)
    return float(means.max() - means.min())


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def save_rc(path, rc):
    np.savez_compressed(path, members=rc.members.astype(np.float32),
                        activations=rc.activations.astype(np.float32), support=rc.support,
                        chain_support=rc.chain_support,
                        member_rhat=rc.member_rhat.astype(np.float32),
                        activation_rhat=rc.activation_rhat.astype(np.float32))


def load_rc(path):
    z = np.load(path)
    return RobustComponents(z["members"].astype(float), z["activations"].astype(float),
                            z["support"], z["chain_support"], z["member_rhat"],
                            z["activation_rhat"])


def write_LA(rc, D, names, prefix):
    pd.DataFrame(rc.members.T, index=D["genes"], columns=names).round(4).to_csv(
        os.path.join(OUT, f"{prefix}L.csv.gz"), index_label="gene")
    pd.DataFrame(rc.activations.T, index=names, columns=D["genomes"]).round(4).to_csv(
        os.path.join(OUT, f"{prefix}A.csv.gz"), index_label="phylon")


def common_params():
    return dict(likelihood_power=ARGS.zeta, detection_effects=("sample",),
                background_effects=("sample",), n_chains=ARGS.chains,
                store_draws=ARGS.store_draws, min_support=MIN_SIZE,
                checkpoint_every=ARGS.checkpoint_every)


# ----------------------------------------------------------------------------- stage 1
def discovery(D):
    path = os.path.join(OUT, "stage1")
    os.makedirs(path, exist_ok=True)
    done = os.path.join(path, "summary.json")
    if os.path.exists(done):
        log("discovery: already done, loading", path)
        return load_rc(os.path.join(path, "robust.npz")), json.load(open(done))
    coding = np.where(D["absence"], "absence", "presence")
    starts_file = os.path.join(path, "starts.npz")
    if os.path.exists(starts_file):
        z = np.load(starts_file)
        starts = [(z[f"m{c}"], z[f"a{c}"]) for c in range(ARGS.chains)]
    else:
        from boolmf.utils.validation import to_binary_int8

        V = to_binary_int8(D["Xm"].astype(float), mask=D["mask"])
        starts = []
        for c in range(ARGS.chains):
            t = time.time()
            est = BayesianBooleanMF(init="nmf", init_params={
                "n_components": ARGS.init_rank, "init": "random", "random_state": c,
                "max_iter": 1000})
            starts.append(est._initial_state(V, ARGS.max_components))
            log(f"NMF start {c}: {starts[-1][0].shape[0]} components in "
                f"{time.time() - t:.0f} s")
        np.savez_compressed(starts_file, **{f"m{c}": s[0] for c, s in enumerate(starts)},
                            **{f"a{c}": s[1] for c, s in enumerate(starts)})
    burn = ARGS.sweeps // 2
    m = BayesianBooleanMF(max_components=ARGS.max_components, burn_in=burn,
                          n_draws=max(1, (ARGS.sweeps - burn) // 5), thin=5,
                          max_sweeps=ARGS.sweeps, random_state=100, init=starts,
                          population_moves=ARGS.population_every, feature_groups=coding,
                          checkpoint_dir=os.path.join(OUT, "checkpoints", "stage1"),
                          **common_params())
    log(f"discovery: {ARGS.chains} chains x {ARGS.sweeps} sweeps, {ARGS.threads} threads")
    w = watch(os.path.join(OUT, "checkpoints", "stage1"), "discovery", ARGS.sweeps)
    t = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(D["Xm"].astype(float), mask=D["mask"])
    w.stop.set()
    seconds = time.time() - t
    for x in caught:
        log("warning:", x.message)
    log(f"discovery: done in {seconds / 3600:.2f} h; finding robust components")
    rc = m.robust_components()
    stability = m.robust_stability(n_windows=4)
    draws = m._chain_draws()
    sizes = [int(((Z.sum(0) >= MIN_SIZE) & (U.sum(0) >= MIN_SIZE)).sum())
             for chain in draws for Z, U in (chain[t] for t in range(len(chain)))]
    transient = float(np.mean(sizes) - rc.support.sum())
    conv = rc.convergence()
    conv_comp = conv.pop("components")
    det, bg = per_cell_rates(m, coding)
    summary = {
        "seconds": seconds, "n_robust": rc.n_components,
        "transient_per_draw": transient, "rhat": m.rhat_,
        "log_likelihood_chain_spread": chain_spread(m),
        "population_acceptance": m.population_acceptance_,
        "split_merge_acceptance": m.split_merge_acceptance_,
        "stability": stability, "entry_convergence": conv,
        "entry_convergence_core_margin": core_margin(rc),
        "components_q99_rhat": conv_comp,
        "heldout_full_posterior": heldout_ll(m.predictive_probability(), D),
        "heldout_robust_LA": heldout_ll(rc.reconstruct(det, bg), D),
        "precision_recall": precision_recall(rc, D),
        "chains": m.chain_status_.tolist(),
    }
    save_rc(os.path.join(path, "robust.npz"), rc)
    json.dump(jsonable(summary), open(done, "w"), indent=1)
    log(f"discovery: {rc.n_components} robust components, {transient:.0f} other components "
        f"per draw; held-out log-likelihood {summary['heldout_full_posterior']}")
    return rc, jsonable(summary)


# ----------------------------------------------------------------------------- stage 2
def refit(D, rc1, s1):
    coding = np.where(D["absence"], "absence", "presence")
    K = rc1.n_components
    if ARGS.free_slots == "auto":
        free = int(max(100, np.ceil(2 * s1["transient_per_draw"])))
    else:
        free = int(ARGS.free_slots)
    starts = [rc1.sample(random_state=1000 + c) for c in range(ARGS.chains)]
    burn = ARGS.refit_sweeps // 2
    m = BayesianBooleanMF(max_components=K + free, burn_in=burn,
                          n_draws=max(1, (ARGS.refit_sweeps - burn) // 5), thin=5,
                          max_sweeps=ARGS.refit_sweeps, random_state=200, init=starts,
                          pin_init=True, feature_groups=coding, n_jobs=ARGS.chains,
                          checkpoint_dir=os.path.join(OUT, "checkpoints", "stage2"),
                          **common_params())
    log(f"refit: {K} pinned components + {free} free slots, {ARGS.chains} independent chains "
        f"x {ARGS.refit_sweeps} sweeps")
    w = watch(os.path.join(OUT, "checkpoints", "stage2"), "refit", ARGS.refit_sweeps)
    t = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(D["Xm"].astype(float), mask=D["mask"])
    w.stop.set()
    seconds = time.time() - t
    for x in caught:
        log("warning:", x.message)
    log(f"refit: done in {seconds / 3600:.2f} h")
    pc = m.pinned_components()
    conv = pc.convergence()
    comp_rhat = conv.pop("components")
    # robust components of the refit found by matching over all slots (pinned and free): the
    # same summary as the discovery stage, from independent chains
    allrc = m.robust_components()
    conv_all = allrc.convergence()
    conv_all_comp = conv_all.pop("components")
    Rp, Gp = (x.astype(np.float32) for x in (pc.majority()[1].T, pc.majority()[0]))
    Ra, Ga = (x.astype(np.float32) for x in (allrc.majority()[1].T, allrc.majority()[0]))
    new = int((_cell_jaccard(Ra, Ga, Rp, Gp).max(1) < 0.5).sum()) if len(Ra) and len(Rp) \
        else len(Ra)
    det, bg = per_cell_rates(m, coding)
    summary = {
        "seconds": seconds, "n_pinned": K, "free_slots": free, "rhat": m.rhat_,
        "log_likelihood_chain_spread": chain_spread(m),
        "split_merge_acceptance": m.split_merge_acceptance_,
        "pinned_support_min": float(pc.support.min()) if K else None,
        "pinned_support_below_0.5": int((pc.support < 0.5).sum()),
        "entry_convergence": conv,
        "entry_convergence_core_margin": core_margin(pc),
        "matched_components": allrc.n_components,
        "matched_entry_convergence": conv_all,
        "matched_entry_convergence_core_margin": core_margin(allrc),
        "matched_components_q99_rhat_below_1.1": float(np.mean(conv_all_comp < 1.1))
        if allrc.n_components else None,
        "components_q99_rhat_below": {t_: float(np.mean(comp_rhat < t_))
                                      for t_ in (1.01, 1.05, 1.1, 1.2)},
        "robust_in_free_slots_unmatched": new,
        "heldout_full_posterior": heldout_ll(m.predictive_probability(), D),
        "heldout_phylon_LA": heldout_ll(pc.reconstruct(det, bg), D),
        "precision_recall": precision_recall(pc, D),
        "chains": m.chain_status_.tolist(),
    }
    return m, pc, comp_rhat, jsonable(summary)


# ----------------------------------------------------------------------------- main
def gene_lists(L, k, genes, mask, top=50):
    idx = np.flatnonzero((L[k] >= 0.5) & mask)
    idx = idx[np.argsort(-L[k, idx], kind="stable")][:top]
    return ";".join(genes[idx])


def main():
    log("args:", vars(ARGS))
    D = prepare()
    np.save(os.path.join(OUT, "heldout.npy"), D["test"].T)
    rc1, s1 = discovery(D)
    names1 = [f"stage1_{k + 1}" for k in range(rc1.n_components)]
    write_LA(rc1, D, names1, "stage1_")
    if rc1.n_components == 0:
        log("no robust components; stopping after the discovery stage")
        json.dump({"stage1": s1}, open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1)
        return
    m, pc, comp_rhat, s2 = refit(D, rc1, s1)
    names = [f"phylon_{k + 1}" for k in range(pc.n_components)]
    write_LA(pc, D, names, "")
    pres, absn = ~D["absence"], D["absence"]
    Lp, Ap = pc.members, pc.activations
    phy = pd.DataFrame({
        "phylon": names,
        "stage1_support": rc1.support,
        "stage1_chains_present": (rc1.chain_support > 0).sum(1),
        "support": pc.support,
        "supported_in_refit": pc.support >= 0.5,
        "n_genes_present": ((Lp >= 0.5) & pres).sum(1),
        "n_genes_absent": ((Lp >= 0.5) & absn).sum(1),
        "n_genomes": (Ap >= 0.5).sum(0),
        "expected_genes": Lp.sum(1),
        "expected_genomes": Ap.sum(0),
        "rhat_q99": comp_rhat,
        "genes_present_top50": [gene_lists(Lp, k, D["genes"], pres) for k in range(len(names))],
        "genes_absent_top50": [gene_lists(Lp, k, D["genes"], absn) for k in range(len(names))],
    })
    phy.to_csv(os.path.join(OUT, "phylons.csv"), index=False)
    pd.DataFrame({"gene": D["genes"], "frequency": D["freq"],
                  "coding": np.where(D["absence"], "absence", "presence"),
                  "n_phylons": (Lp >= 0.5).sum(0)}).to_csv(os.path.join(OUT, "genes.csv"),
                                                          index=False)
    labels = list(m.feature_groups_)
    g = pd.DataFrame({"genome": D["genomes"], "n_phylons": (Ap >= 0.5).sum(1)})
    for lab in labels:
        i = labels.index(lab)
        g[f"detection_{lab}_coded"] = m.detection_rate_per_sample_[:, i]
        g[f"background_{lab}_coded"] = m.background_rate_per_sample_[:, i]
    g.to_csv(os.path.join(OUT, "genomes.csv"), index=False)
    pred = m.predictive_probability()                            # minority-coded scale
    pred = np.where(D["absence"][None, :], 1.0 - pred, pred)     # P(gene present)
    np.save(os.path.join(OUT, "predictive.npy"), pred.T.astype(np.float32))
    diag = {"input": {"genes": int(len(D["genes"])), "genomes": int(len(D["genomes"])),
                      "presence_coded": int(pres.sum()), "absence_coded": int(absn.sum()),
                      "heldout_cells": int(D["test"].sum())},
            "args": vars(ARGS), "stage1": s1, "stage2": s2,
            "minutes_total": (time.time() - T0) / 60}
    json.dump(jsonable(diag), open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1)
    c = s2["entry_convergence"]
    log(f"done: {pc.n_components} phylons; R-hat < 1.1 for "
        f"{c['members']['rhat_below']['1.1']:.1%} of membership and "
        f"{c['activations']['rhat_below']['1.1']:.1%} of activation entries; held-out "
        f"log-likelihood {s2['heldout_full_posterior']}, precision/recall "
        f"{s2['precision_recall']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # keep the reason in the log too
        log("failed:", repr(e))
        raise
    sys.exit(0)
