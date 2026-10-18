"""Choose ζ (``likelihood_power``) for BBMF phylons: one discovery fit per ζ on the same
held-out cells, and one comparison table.

Every ζ runs ``bbmf_phylons.py --skip-refit`` into out_dir/zeta_<ζ> with the same holdout
(validation and test cells, from --holdout, --test-share and --seed), so all runs are scored on
identical cells. Choose ζ on the validation cells and report the test cells; then fit the final
phylons on all cells at that ζ (bbmf_phylons.py --holdout 0). Finished runs are skipped and
interrupted ones resume, so rerunning the same command continues the sweep.

Outputs (in out_dir)
--------------------
zeta_sweep.csv          one row per ζ (and one for --compare-L/-A phylons): robust components,
                        components per draw, held-out log-likelihood (full posterior and the OR
                        of the robust L and A) on validation and test cells with genome-resampled
                        95% intervals, precision / recall / F1, R-hat of the rates and the
                        log-likelihood, robust-set stability, hours
zeta_persistence.csv    share of the robust components of the row's ζ found again (majority
                        sets, cell Jaccard >= 0.5) among those of the column's ζ
zeta_sweep.txt          the two tables as text

How to read it: the held-out log-likelihood usually keeps rising with ζ, because more
components fit more detail; coarsening is there for robustness, not to maximize the fit. Look
for the ζ beyond which the validation log-likelihood gains little (Miller & Dunson 2019 choose
it from where the fit stops improving steeply), and check that its robust components persist
at neighbouring ζ.

usage: python bbmf_zeta_sweep.py P.pickle out_dir --zetas 0.2 0.35 0.5 0.8 --threads 32 \\
           [--jobs 1] [-- more bbmf_phylons.py options, e.g. --compare-L L.csv ...]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "bbmf_phylons.py")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("P", help="P matrix file (genes x genomes)")
    p.add_argument("out", help="output directory (one subdirectory per ζ)")
    p.add_argument("--zetas", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.8])
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="cores in total")
    p.add_argument("--jobs", type=int, default=1,
                   help="runs at the same time (each gets threads / jobs cores)")
    p.add_argument("--summarize-only", action="store_true",
                   help="only rebuild the tables from finished runs")
    p.epilog = "Options after -- are passed to every bbmf_phylons.py run."
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    a.rest = rest
    for bad in ("--zeta", "--skip-refit", "--threads"):
        if any(x == bad or x.startswith(bad + "=") for x in a.rest):
            p.error(f"{bad} is set by the sweep; leave it out of the passed options")
    return a


def run_dir(out, z):
    return os.path.join(out, f"zeta_{z:g}")


def finished(d):
    f = os.path.join(d, "diagnostics.json")
    if not os.path.exists(f):
        return False
    try:
        return json.load(open(f)).get("stage1") is not None
    except ValueError:
        return False


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def run_all(a, fh):
    todo = [z for z in a.zetas if not finished(run_dir(a.out, z))]
    for z in a.zetas:
        if z not in todo:
            log(f"zeta {z:g}: already finished", fh)
    threads = max(1, a.threads // max(1, a.jobs))
    running = {}
    while todo or running:
        while todo and len(running) < max(1, a.jobs):
            z = todo.pop(0)
            d = run_dir(a.out, z)
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, SCRIPT, a.P, d, "--zeta", f"{z:g}", "--skip-refit",
                   "--threads", str(threads)] + a.rest
            log(f"zeta {z:g}: start ({threads} threads): {' '.join(cmd)}", fh)
            out = open(os.path.join(d, "run.out"), "a")
            running[z] = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), out,
                          time.time())
        time.sleep(30)
        for z, (proc, out, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            out.close()
            del running[z]
            hours = (time.time() - t0) / 3600
            if proc.returncode == 0:
                log(f"zeta {z:g}: finished in {hours:.2f} h", fh)
            else:
                log(f"zeta {z:g}: FAILED (exit {proc.returncode}) after {hours:.2f} h; see "
                    f"{os.path.join(run_dir(a.out, z), 'run.out')}", fh)


# ----------------------------------------------------------------------------- tables
def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def _metrics(prefix, ev):
    row = {}
    for part in ("validation", "test"):
        r = _get(ev, part) or {}
        ci = r.get("log_likelihood_95") or [None, None]
        row[f"{part}_{prefix}_ll"] = r.get("log_likelihood")
        row[f"{part}_{prefix}_ll_lo"], row[f"{part}_{prefix}_ll_hi"] = ci
        if "F1" in r:
            row[f"{part}_precision"] = r.get("precision")
            row[f"{part}_recall"] = r.get("recall")
            row[f"{part}_F1"] = r.get("F1")
            f1ci = r.get("F1_95") or [None, None]
            row[f"{part}_F1_lo"], row[f"{part}_F1_hi"] = f1ci
    return row


def _majority(path):
    z = np.load(path)
    return ((z["activations"] >= 0.5).T.astype(np.float32),
            (z["members"] >= 0.5).astype(np.float32))


def _cell_jaccard(Ra, Ga, Rb, Gb):
    inter = (Ra @ Rb.T) * (Ga @ Gb.T)
    size_a = Ra.sum(1) * Ga.sum(1)
    size_b = Rb.sum(1) * Gb.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(inter > 0, inter / (size_a[:, None] + size_b[None, :] - inter), 0.0)


def summarize(a, fh):
    rows, sets, comparison = [], {}, None
    for z in sorted(a.zetas):
        d = run_dir(a.out, z)
        if not finished(d):
            log(f"zeta {z:g}: not finished, left out of the tables", fh)
            continue
        diag = json.load(open(os.path.join(d, "diagnostics.json")))
        s1 = diag["stage1"]
        stab = s1.get("stability") or []
        row = {"zeta": z, "robust_components": s1["n_robust"],
               "other_components_per_draw": s1["transient_per_draw"],
               "slots": s1.get("slots"), "slots_used_median": s1.get("slots_used_median")}
        row.update(_metrics("full", s1.get("heldout_full_posterior")))
        row.update(_metrics("LA", s1.get("heldout_robust_LA")))
        row.update({"rhat_detection": _get(s1, "rhat", "detection_rate"),
                    "rhat_background": _get(s1, "rhat", "background_rate"),
                    "rhat_log_likelihood": _get(s1, "rhat", "log_likelihood"),
                    "stability_first_window": stab[0]["share_of_last"] if stab else None,
                    "hours": s1.get("seconds", 0) / 3600})
        rows.append(row)
        sets[z] = _majority(os.path.join(d, "stage1", "robust.npz"))
        if comparison is None and diag.get("comparison"):
            comparison = diag["comparison"]
    if not rows:
        log("no finished runs yet", fh)
        return
    tab = pd.DataFrame(rows)
    tab["validation_full_ll_gain"] = tab["validation_full_ll"].diff()
    if comparison:
        crow = {"zeta": np.nan, "robust_components": comparison["phylons"]}
        crow.update(_metrics("LA", comparison["evaluation"]))
        crow = pd.DataFrame([crow], index=[comparison["name"]])
        tab = pd.concat([tab.set_index(tab["zeta"].map(lambda v: f"zeta {v:g}")), crow])
    else:
        tab = tab.set_index(tab["zeta"].map(lambda v: f"zeta {v:g}"))
    zs = sorted(sets)
    per = pd.DataFrame(index=[f"zeta {z:g}" for z in zs], columns=[f"zeta {z:g}" for z in zs],
                       dtype=float)
    for zi in zs:
        Ri, Gi = sets[zi]
        for zj in zs:
            Rj, Gj = sets[zj]
            if not len(Ri):
                continue
            hit = (_cell_jaccard(Ri, Gi, Rj, Gj).max(1) >= 0.5) if len(Rj) else \
                np.zeros(len(Ri), bool)
            per.loc[f"zeta {zi:g}", f"zeta {zj:g}"] = float(hit.mean())
    tab.to_csv(os.path.join(a.out, "zeta_sweep.csv"), index_label="run")
    per.to_csv(os.path.join(a.out, "zeta_persistence.csv"), index_label="robust_of")
    show = [c for c in ("robust_components", "other_components_per_draw", "validation_full_ll",
                        "validation_LA_ll", "validation_F1", "test_full_ll", "test_LA_ll",
                        "test_precision", "test_recall", "test_F1", "rhat_detection",
                        "rhat_background", "stability_first_window", "hours")
            if c in tab.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        text = (tab[show].round(4).to_string() + "\n\npersistence (row's robust components "
                "found among the column's):\n" + per.round(3).to_string() + "\n")
    open(os.path.join(a.out, "zeta_sweep.txt"), "w").write(text)
    print(text, flush=True)
    log(f"tables written to {a.out}", fh)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, "sweep_log.txt"), "a")
    log(f"zeta sweep: {a.zetas}, {a.threads} threads, {a.jobs} at a time; passed options: "
        f"{a.rest}", fh)
    if not a.summarize_only:
        run_all(a, fh)
    summarize(a, fh)


if __name__ == "__main__":
    main()
