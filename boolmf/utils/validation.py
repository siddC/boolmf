"""Input validation helpers."""

import hashlib

import numpy as np
import scipy.sparse as sp


def to_binary_int8(X, mask=None, binarize=None):
    """Convert validated input to int8 with 1 = present, 0 = absent, -1 = missing.

    Parameters
    ----------
    X : ndarray or sparse matrix (already passed through ``validate_data``)
    mask : array-like of bool, optional
        True marks entries to ignore (``numpy.ma`` convention).
    binarize : float or None
        None requires 0/1 input (NaN allowed); a float maps values above it to 1.
    """
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype == bool:
        V = X.astype(np.int8)
        missing = np.zeros(X.shape, bool)
    else:
        missing = np.isnan(X) if np.issubdtype(X.dtype, np.floating) else np.zeros(X.shape, bool)
        if binarize is None:
            ok = missing | (X == 0) | (X == 1)
            if not ok.all():
                raise ValueError(
                    "BoolMF expects binary input (0/1, NaN for missing). "
                    "Pass binarize=<threshold> to binarize other values."
                )
            V = np.where(missing, 0, X).astype(np.int8)
        else:
            with np.errstate(invalid="ignore"):
                V = (X > binarize).astype(np.int8)
    V[missing] = -1
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != V.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {V.shape}.")
        V[mask] = -1
    return V


def matrix_fingerprint(V):
    """Stable 64-bit digest of an int8 matrix (used to recognize the training data)."""
    h = hashlib.blake2b(np.ascontiguousarray(V).tobytes(), digest_size=16)
    h.update(str(V.shape).encode())
    return h.hexdigest()


def row_seeds(V, base_seed):
    """One 64-bit seed per row, derived from the row's content and ``base_seed``."""
    seeds = np.empty(V.shape[0], np.uint64)
    salt = int(base_seed).to_bytes(8, "little", signed=False)
    for i in range(V.shape[0]):
        d = hashlib.blake2b(np.ascontiguousarray(V[i]).tobytes(), digest_size=8, key=salt)
        seeds[i] = np.frombuffer(d.digest(), dtype=np.uint64)[0]
    return seeds
