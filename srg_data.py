"""
A WL-defeating strongly-regular-graph pair: the 4x4 Rook's graph vs the
Shrikhande graph. Both are SRG(16,6,2,2), regular and cospectral, so the
walk/reachability covers (and the WL test) cannot tell them apart. They differ
only in local arrangement -- each vertex's neighbourhood induces two triangles
(Rook) or a 6-cycle (Shrikhande) -- which the sieve cover detects.

The dataset is random relabellings of the two graphs; the task is to tell which
graph a sample is. Balanced classes (accuracy is the standard metric for this
isomorphism-discrimination setting).
"""
from __future__ import annotations
import numpy as np


def rook44():
    idx = [(i, j) for i in range(4) for j in range(4)]
    n = 16; A = np.zeros((n, n), np.float32)
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            if a != b and (i == k or j == l):
                A[a, b] = 1.0
    return A


def shrikhande():
    S = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    idx = [(i, j) for i in range(4) for j in range(4)]
    n = 16; A = np.zeros((n, n), np.float32)
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            if a != b and ((k - i) % 4, (l - j) % 4) in S:
                A[a, b] = 1.0
    return A


def _permute(A, rng):
    p = rng.permutation(A.shape[0])
    return A[np.ix_(p, p)].astype(np.float32)


def make_dataset(n_per_class=300, seed=0):
    rng = np.random.default_rng(seed)
    base = {0: rook44(), 1: shrikhande()}
    As, ys = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            As.append(_permute(base[label], rng)); ys.append(label)
    order = rng.permutation(len(ys))
    return [As[i] for i in order], np.asarray(ys, np.int64)[order]
