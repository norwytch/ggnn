"""
Loader for the Platonov et al. (2023) corrected heterophily datasets, the calibration
set for palimpsest. Pulls the .npz directly from the authors' repo so there is no
torch-geometric dependency, and exposes features, labels, edges, and the 10 official
train/val/test splits.

  https://github.com/yandex-research/heterophilous-graphs

`synthetic_heterophily` fabricates a tiny featureful heterophilic graph so the harness
and the smoke test run without a download.
"""
from __future__ import annotations
import os
import urllib.request

import numpy as np

_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{}.npz"
_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "hetero")

# name -> leaderboard metric
DATASETS = {
    "roman_empire": "accuracy",
    "amazon_ratings": "accuracy",
    "minesweeper": "auroc",
    "tolokers": "auroc",
    "questions": "auroc",
}


def load(name):
    """Download (once, cached) and load a Platonov dataset. Returns a dict with x, y,
    edges (undirected, both directions added), the 10 split masks, and the metric."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; options: {list(DATASETS)}")
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"{name}.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_URL.format(name), path)
    d = np.load(path)
    edges = d["edges"].astype(np.int64)
    edges = np.concatenate([edges, edges[:, ::-1]], 0)          # symmetrize
    return {
        "x": d["node_features"].astype(np.float32),
        "y": d["node_labels"].astype(np.int64),
        "edges": edges,                                          # (2E, 2)
        "train": d["train_masks"].astype(bool),                 # (10, n)
        "val": d["val_masks"].astype(bool),
        "test": d["test_masks"].astype(bool),
        "metric": DATASETS[name],
        "n_classes": int(d["node_labels"].max()) + 1,
    }


def synthetic_heterophily(n=400, d_feat=16, n_classes=4, seed=0):
    """A small heterophilic graph: nodes wire preferentially to OTHER classes, and each
    class has a distinct feature mean. A model that can use features to invert/discount
    cross-class edges (a sheaf) has something to learn from; plain smoothing is hurt by
    the heterophily. For plumbing and the smoke test, not a benchmark."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    means = rng.standard_normal((n_classes, d_feat)) * 2.0
    x = (means[y] + rng.standard_normal((n, d_feat))).astype(np.float32)
    src, dst = [], []
    for v in range(n):
        for _ in range(5):                                       # ~5 edges/node
            u = int(rng.integers(0, n))
            if rng.random() < 0.8:                               # 80% cross-class
                others = np.where(y != y[v])[0]
                u = int(rng.choice(others))
            src += [v, u]; dst += [u, v]
    edges = np.stack([np.array(src), np.array(dst)], 1).astype(np.int64)
    masks = np.zeros((1, n), bool)
    idx = rng.permutation(n)
    tr, va = idx[: n // 2], idx[n // 2: 3 * n // 4]
    te = idx[3 * n // 4:]
    train = np.zeros((1, n), bool); train[0, tr] = True
    val = np.zeros((1, n), bool); val[0, va] = True
    test = np.zeros((1, n), bool); test[0, te] = True
    return {"x": x, "y": y, "edges": edges, "train": train, "val": val,
            "test": test, "metric": "accuracy", "n_classes": n_classes}
