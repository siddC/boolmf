"""Public datasets used by the reproductions, downloaded once from the PMLB repository on GitHub
(Romano et al. 2021, https://github.com/EpistasisLab/pmlb; UCI originals, CC BY 4.0)."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / ".cache"
URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"


def pmlb(name):
    """The PMLB table ``name`` as a DataFrame (the class is the ``target`` column)."""
    path = CACHE / f"{name}.tsv.gz"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path)
    return pd.read_csv(path, sep="\t")


def one_hot(name):
    """Every attribute value, class included, as a binary column (pandas.get_dummies order):
    Mushroom 8124 x 119, Tic-tac-toe 958 x 29, Chess 3196 x 75."""
    return pd.get_dummies(pmlb(name).astype(str)).to_numpy(bool)


def digits():
    """UCI Multiple Features pixel data, 2000 x 240, pixel > 0 as 1 (291,654 ones, as in
    Miettinen et al. 2008, Table 2)."""
    return pmlb("mfeat_pixel").drop(columns="target").to_numpy() > 0


def coverage_curve(X, usage, basis):
    """c(k) = 1 - E_k / |X| after the first k components, E_k the Hamming distance between X
    and the Boolean product (Belohlavek & Trnecka 2015, Eq. 10)."""
    covered = np.zeros(X.shape, bool)
    total = X.sum()
    out = []
    for k in range(basis.shape[0]):
        covered |= np.outer(usage[:, k], basis[k])
        out.append(1.0 - np.count_nonzero(covered != X) / total)
    return np.array(out)


def factors_needed(curve, levels=(0.25, 0.5, 0.75, 0.95, 1.0)):
    """Smallest number of components reaching each coverage level (None if never)."""
    return [int(np.argmax(curve >= c - 1e-12)) + 1 if (curve >= c - 1e-12).any() else None
            for c in levels]
