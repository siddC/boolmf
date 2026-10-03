"""v0.2 gate on real data: the GENOMiCUS-4k E. coli pangenome (3,944 genomes).

The data are not distributed with boolmf; pass the gene x genome presence/absence matrices
(parquet, genes as rows, genomes as columns). The K-12 MG1655 reference appears as "511145.12"
in some matrices and "sequence" in others; both are mapped to "sequence".

Subsample: ``--genomes`` random genomes (seed 0); accessory genes present in 1-99% of them.
10% of the entries (stratified by presence) are held out. Compared on the held-out entries
(mean log-likelihood, AUC, accuracy at 0.5):

* NMF: Frobenius NMF fit on the training entries only (weighted multiplicative updates,
  NNDSVDa start), rank chosen from 10, 20, 40, 80, 160 by the log-likelihood on a further 10%
  of the training entries; probabilities from the reconstruction clipped to [eps, 1 - eps]
  (eps fit on the training entries) and, separately, by isotonic regression fit on the
  validation entries.
* BayesianBooleanMF: 4 chains, 3,000 burn-in sweeps and 1,000 kept, ``--variant``:
  ``ibp`` (defaults, Indian buffet process), ``alpha10`` (alpha fixed at 10) or ``sample``
  (per-genome detection and background rates).

With ``--core`` the core genes are added and the inner core (genes in at least 99% of all
genomes) becomes an anchor component with fixed members (``--variant sample`` recommended); the
soft core stays free. A learned anchor absorbs the soft core and about a quarter of the
accessory genes, and predicts the accessory genes worse.

Results (held-out log-likelihood / AUC / accuracy; R-hat of the detection and background
rates; components per draw):

500 genomes x 4,811 accessory genes
  NMF rank 80, clipped             -0.157 / 0.984 / 0.951
  NMF rank 80, isotonic            -0.148 / 0.982 / 0.950
  ibp (900 slots)                  -0.115 / 0.986 / 0.966   R-hat 2.84 / 1.74   ~690
  n_components=80 (fixed rank)     -0.198 / 0.956 / 0.943
200 genomes x 4,810 accessory genes
  NMF rank 20, clipped             -0.205 / 0.972 / 0.927
  NMF rank 20, isotonic            -0.199 / 0.972 / 0.926
  ibp (900 slots)                  -0.138 / 0.983 / 0.957   R-hat 2.22 / 1.45   ~470
  alpha10 (300 slots)              -0.138 / 0.984 / 0.956   R-hat 2.78 / 2.06   ~250 (cap)
  sample (600 slots)               -0.123 / 0.988 / 0.960   R-hat 1.18 / 1.07   ~485
200 genomes x 7,103 genes (core + accessory), sample, 600 slots
  inner core as a fixed anchor     -0.093 / 0.992 / 0.972   R-hat 1.18 / 1.14
    accessory entries only         -0.115 / 0.989 / 0.963
  inner core as a learned anchor   -0.138 / 0.980 / 0.959   R-hat 1.11 / 1.02
    accessory entries only         -0.168 / 0.978 / 0.944

The Bayesian model predicts held-out entries better than NMF in every Indian buffet process
variant, but no run converges: the log-likelihood R-hat is about 3 in all of them and its
effective sample size about 5. After 4,000 sweeps the chains are still adding components and
raising the log-likelihood; the data support hundreds of components (more than the number of
genomes at 200), as gene gains along a phylogeny would. With the fixed anchor most soft-core
genes (610 of 724) go to two components carried by 88% and 59% of the genomes.

Phylogroup-balanced subsample (``--metadata``; 249 genomes x 4,811 accessory genes), per-genome
rates, 4 chains of 20,000 sweeps (``--sweeps 20000 --variant sample``):
  NMF rank 20, isotonic            -0.187 / 0.974 / 0.931
  BayesianBooleanMF, each chain    -0.119 to -0.122 / 0.987 / 0.961   pooled -0.106 / 0.991 / 0.966
  It does not settle: from sweep 10,000 to 20,000 the log-likelihood rises by about 650 per
  1,000 sweeps in every chain and the component count by about 3 (to about 610); over the last
  5,000 sweeps R-hat is 2.8 for the log-likelihood, 1.45 for the component count and 1.14 / 1.07
  for the rates. Chains agree on prediction (mean absolute difference of held-out predictive
  probabilities 0.028; per-genome detection rates correlate at 0.95) but not on components:
  only 14% of one chain's components (2% of the entries they cover) have a partner in another
  chain with member and carrier Jaccard >= 0.8. The same entries are covered by different
  sets of overlapping components.

Run: python benchmarks/genomicus4k.py --accessory P_acc.parquet [--core P_core.parquet]
     [--genomes 500 | --metadata metadata.parquet] [--variant ibp] [--sweeps 4000]
     [--checkpoint-dir DIR] [--nmf-only]
"""
import argparse
import time
import warnings

import numpy as np

from boolmf import AnchorComponent, BayesianBooleanMF
from boolmf.model_selection import EntryShuffleSplit

ALIASES = {"511145.12": "sequence"}
RANKS = (10, 20, 40, 80, 160)
VARIANTS = {
    "ibp": {},
    "alpha10": {"alpha_prior": 10.0},
    "sample": {"detection_effects": ("sample",), "background_effects": ("sample",)},
}


def load(path, genomes=None):
    import pandas as pd

    df = pd.read_parquet(path).rename(columns=ALIASES)
    if genomes is not None:
        df = df[genomes]
    return df


def masked_nmf(X, train, k, n_iter=500, seed=0, tol=1e-5):
    """Frobenius NMF on the entries where ``train`` is True."""
    from sklearn.decomposition._nmf import _initialize_nmf

    Xf = np.where(train, X, 0.0)
    fill = np.where(train, X, (Xf.sum(0) / np.maximum(train.sum(0), 1))[None, :])
    W, H = _initialize_nmf(fill, k, init="nndsvda", random_state=seed)
    M = train.astype(float)
    prev = np.inf
    for it in range(n_iter):
        H *= (W.T @ Xf) / np.maximum(W.T @ (M * (W @ H)), 1e-12)
        W *= (Xf @ H.T) / np.maximum((M * (W @ H)) @ H.T, 1e-12)
        if it % 20 == 0:
            err = float(((M * (W @ H - Xf)) ** 2).sum())
            if prev - err < tol * prev:
                break
            prev = err
    return W @ H


def clip_calibrate(R, X, train):
    best = None
    for eps in np.logspace(-6, -0.5, 60):
        q = np.clip(R[train], eps, 1 - eps)
        ll = np.mean(np.where(X[train] == 1, np.log(q), np.log1p(-q)))
        if best is None or ll > best[0]:
            best = (ll, eps)
    return np.clip(R, best[1], 1 - best[1])


def scores(P, X, entries):
    """Mean log-likelihood, AUC and accuracy at 0.5 on ``entries``."""
    from sklearn.metrics import roc_auc_score

    p, x = np.clip(P[entries], 1e-12, 1 - 1e-12), X[entries]
    ll = float(np.mean(np.where(x == 1, np.log(p), np.log1p(-p))))
    return ll, float(roc_auc_score(x, p)), float(np.mean((p >= 0.5) == (x == 1)))


def nmf_baseline(X, test):
    from sklearn.isotonic import IsotonicRegression

    train = ~test
    val = np.zeros_like(test)
    val[train] = np.random.default_rng(1).random(train.sum()) < 0.1
    inner = train & ~val
    by_rank = {}
    for k in RANKS:
        R = masked_nmf(X, inner, k)
        by_rank[k] = (scores(clip_calibrate(R, X, inner), X, val)[0], R)
    k = max(by_rank, key=lambda r: by_rank[r][0])
    R_inner = by_rank[k][1]
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    iso.fit(R_inner[val], X[val])
    isotonic = scores(iso.predict(R_inner.ravel()).reshape(X.shape), X, test)
    clipped = scores(clip_calibrate(masked_nmf(X, train, k), X, train), X, test)
    return k, clipped, isotonic


# phylogroup-balanced subsample: 25 per Clermont phylogroup, 10 per Shigella species, clade I
QUOTA = {**{g: 25 for g in ("A", "B1", "B2", "C", "D", "E", "F", "G")},
         **{f"Shigella {s}": 10 for s in ("flexneri", "sonnei", "dysenteriae", "boydii")},
         "cladeI": 9}


def balanced_subsample(genomes, metadata, rng):
    """Column indices of the phylogroup-balanced subsample (metadata only picks genomes; the
    model sees nothing but the presence/absence matrix)."""
    import pandas as pd

    md = pd.read_parquet(metadata)
    md["gid"] = md["genome_id"].astype(str).replace(ALIASES)
    group = md.drop_duplicates("gid").set_index("gid").reindex(list(genomes))["phylogroup"]
    group = group.to_numpy()
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(group == g), q, replace=False)
                                   for g, q in QUOTA.items()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accessory", required=True)
    ap.add_argument("--core")
    ap.add_argument("--genomes", type=int, default=500)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="ibp")
    ap.add_argument("--max-components", type=int, default=900)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--nmf-only", action="store_true")
    ap.add_argument("--metadata", help="metadata parquet with genome_id and phylogroup columns: "
                    "take the phylogroup-balanced subsample instead of --genomes random ones")
    ap.add_argument("--sweeps", type=int, default=4000)
    ap.add_argument("--checkpoint-dir")
    args = ap.parse_args()

    acc = load(args.accessory)
    rng = np.random.default_rng(0)
    if args.metadata:
        cols = balanced_subsample(acc.columns, args.metadata, rng)
    else:
        cols = np.sort(rng.choice(acc.shape[1], args.genomes, replace=False))
    Xa = acc.to_numpy(np.int8).T[cols]
    f = Xa.mean(0)
    X = Xa[:, (f >= 0.01) & (f <= 0.99)]
    anchors = None
    if args.core:
        core = load(args.core, genomes=acc.columns)
        C = core.to_numpy(np.int8).T
        inner = np.zeros(C.shape[1] + X.shape[1], bool)
        inner[:C.shape[1]] = C.mean(0) >= 0.99
        X = np.hstack([C[cols], X])
        anchors = [AnchorComponent(np.flatnonzero(inner), membership="fixed")]
    X = X.astype(float)
    print(f"X: {X.shape[0]} genomes x {X.shape[1]} genes, density {X.mean():.3f}", flush=True)
    test = next(EntryShuffleSplit(test_size=0.1, random_state=0).split(X))

    t = time.time()
    k, clipped, isotonic = nmf_baseline(X, test)
    print(f"NMF rank {k}: clipped ll {clipped[0]:.4f} auc {clipped[1]:.4f} acc {clipped[2]:.4f}; "
          f"isotonic ll {isotonic[0]:.4f} auc {isotonic[1]:.4f} acc {isotonic[2]:.4f} "
          f"({time.time() - t:.0f} s)", flush=True)
    if args.nmf_only:
        return
    t = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        burn = args.sweeps // 20 if args.sweeps > 4000 else 3000 * args.sweeps // 4000
        thin = 10 if args.sweeps > 4000 else 1
        m = BayesianBooleanMF(n_chains=4, n_jobs=args.jobs, max_components=args.max_components,
                              burn_in=burn, n_draws=(args.sweeps - burn) // thin, thin=thin,
                              max_sweeps=args.sweeps, store_draws=20, anchor_components=anchors,
                              random_state=0, checkpoint_dir=args.checkpoint_dir,
                              **VARIANTS[args.variant]).fit(X, mask=test)
    ll, auc, acc_ = scores(m._predictive_, X, test)
    print(f"BayesianBooleanMF ({args.variant}): ll {ll:.4f} auc {auc:.4f} acc {acc_:.4f} "
          f"({time.time() - t:.0f} s)")
    print(f"  components per draw {m.n_components_draws_.mean():.0f} (robust {m.n_components_}), "
          f"detection {m.detection_rate_:.3f}, background {m.background_rate_:.4f}")
    print("  R-hat", {k: round(v, 3) for k, v in m.rhat_.items()})
    print("  ESS", {k: round(v) for k, v in m.ess_.items()})
    gate = all(m.rhat_[k] <= 1.01 for k in ("detection_rate", "background_rate")) \
        and ll >= max(clipped[0], isotonic[0])
    print("gate passed" if gate else "gate not passed")
    for w in sorted({str(c.message) for c in caught}):
        print("  warning:", w[:160])


if __name__ == "__main__":
    main()
