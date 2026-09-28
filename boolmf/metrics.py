"""Fit metrics for BoolMF.

The observed matrix is the ground truth and the prediction is "explained by an active
component", so TP = present and explained, FP = absent but explained, FN = present but not
explained, TN = absent and not explained.
"""

import numpy as np

__all__ = [
    "confusion_table",
    "component_integrity",
    "component_leakage",
    "calibration_curve",
]


def _rates(tp, fp, fn, tn):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "npv": tn / (tn + fn),
        }
    if np.ndim(tp) == 0:                          # overall level: plain Python floats
        out = {k: float(v) for k, v in out.items()}
    return out


def confusion_table(model, X=None, *, level="overall", entries=None, threshold=None):
    """Confusion counts and rates of a fitted BoolMF against the observed matrix.

    Parameters
    ----------
    model : fitted BoolMF
    X : array-like, optional
        None uses the training matrix (exact posterior averages); other samples are projected.
    level : {"overall", "sample", "feature"}
    entries : array-like of bool, optional
        Restrict to these entries (for example entries held out during fitting). Held-out
        entries are evaluated against ``X``, so pass the full matrix.
    threshold : float or None
        None returns posterior-expected counts (no cut-off); a float calls an entry explained
        when P(explained) >= threshold.

    Returns
    -------
    dict of arrays (per sample or feature) or of floats (overall).
    """
    if X is None:
        V, E = model._train_V_, model._explained_.astype(float)
    else:
        V = model._validate_X(X, None, reset=False)
        E = model.explained_probability(X) if not model._is_training(V) else \
            model._explained_.astype(float)
    obs = V >= 0
    if entries is not None:
        obs = obs & np.asarray(entries, bool)
    x = (V == 1) & obs
    nx = (V == 0) & obs
    e = E if threshold is None else (E >= threshold).astype(float)
    e = np.where(obs, e, 0.0)
    ne = np.where(obs, 1.0 - e, 0.0)
    axis = {"overall": None, "sample": 1, "feature": 0}
    if level not in axis:
        raise ValueError("level must be 'overall', 'sample' or 'feature'.")
    ax = axis[level]
    tp = (e * x).sum(axis=ax)
    fp = (e * nx).sum(axis=ax)
    fn = (ne * x).sum(axis=ax)
    tn = (ne * nx).sum(axis=ax)
    return _rates(tp, fp, fn, tn)


def component_integrity(model):
    """Share of member features present in training samples where the component is active.

    Plug-in estimate from posterior means: sum q_ik m_jk x_ij / sum q_ik m_jk over observed
    entries (q = activation probability, m = membership probability).
    """
    V = model._train_V_
    obs = (V >= 0).astype(float)
    x = (V == 1).astype(float)
    M, Q = model.components_, model.activations_
    num = ((x @ M.T) * Q).sum(0)
    den = ((obs @ M.T) * Q).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def component_leakage(model):
    """Share of member features present where the component is inactive and no other
    active component contains them (plug-in estimate from posterior means).

    High leakage suggests missed carriers.
    """
    V = model._train_V_
    obs = (V >= 0).astype(float)
    x = (V == 1).astype(float)
    M, Q = model.components_, model.activations_
    K = M.shape[0]
    log_none = np.zeros(V.shape)
    terms = []
    for k in range(K):
        t = np.log1p(-np.clip(np.outer(Q[:, k], M[k]), 0, 1 - 1e-12))
        terms.append(t)
        log_none += t
    out = np.full(K, np.nan)
    for k in range(K):
        others_none = np.exp(log_none - terms[k])
        w = np.outer(1.0 - Q[:, k], M[k]) * others_none * obs
        d = w.sum()
        if d > 0:
            out[k] = (w * x).sum() / d
    return out


def calibration_curve(model, X=None, *, entries=None, n_bins=10):
    """Predicted presence probability versus observed presence, in equal-width bins.

    Parameters
    ----------
    model : fitted BoolMF
    X : array-like, optional
        None uses the training matrix; pass the full matrix with ``entries`` to check
        entries held out during fitting.
    entries : array-like of bool, optional
    n_bins : int, default=10

    Returns
    -------
    mean_predicted, observed_fraction, counts : ndarrays of length n_bins
    """
    if X is None:
        V = model._train_V_
        P = model._predictive_.astype(float)
    else:
        V = model._validate_X(X, None, reset=False)
        P = model.predictive_probability(X) if not model._is_training(V) else \
            model._predictive_.astype(float)
    sel = V >= 0
    if entries is not None:
        sel &= np.asarray(entries, bool)
    p, y = P[sel], (V[sel] == 1).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.bincount(idx, weights=p, minlength=n_bins) / counts
        obs_frac = np.bincount(idx, weights=y, minlength=n_bins) / counts
    return mean_pred, obs_frac, counts
