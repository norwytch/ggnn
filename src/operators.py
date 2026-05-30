"""
Low-level graph operators shared by the models and covers.

Two kinds of object live here:

  * 1-hop propagators -- the symmetric normalised adjacency a GCN/GIN multiplies
    by. These are *local*: one application mixes a node with its neighbours, and
    no amount of stacking lets a 1-WL-bounded net read a global property like the
    number of connected components.
  * global-reachability read-outs -- ``n_components`` and ``blast_radius``. These
    are functions of the whole graph, exactly the signal 1-hop aggregation cannot
    recover. ``run_experiment`` feeds them to the trivial baseline and to GCNPlus.

``covers.reachability_cover`` is the same blast-radius signal dressed up as a
cover; this module is the plain-numpy version the non-cover code uses.
"""
from __future__ import annotations
import numpy as np


def _symmetrise(A):
    A = np.asarray(A, np.float32)
    return ((A + A.T) > 0).astype(np.float32)


def n_components(A):
    """Number of (weakly) connected components of a 0/1 adjacency matrix.

    1 for a single directed cycle, k for k disjoint cycles -- the scalar the
    LogReg baseline classifies on, and the thing a 1-WL net provably cannot see.
    """
    Asym = _symmetrise(A)
    n = Asym.shape[0]
    seen = np.zeros(n, dtype=bool)
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(Asym[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
    return count


def gcn_norm(A):
    """Symmetric normalised adjacency with self-loops: D^-1/2 (A+I) D^-1/2.

    The standard GCN 1-hop propagator. Both estate classes are 2-regular once
    symmetrised, so this matrix -- and every power of it -- treats every node
    identically; that is the mechanism behind the base-rate scores.
    """
    Ah = _symmetrise(A) + np.eye(A.shape[0], dtype=np.float32)
    d = Ah.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    return (dinv[:, None] * Ah * dinv[None, :]).astype(np.float32)


def reachable_counts(A, K=None):
    """Per-node size of the within-K-hop reachable set (including self).

    K=None means "to convergence" (K = n), i.e. the true reachable set along
    directed edges.
    """
    A = (np.asarray(A) > 0).astype(np.float32)
    n = A.shape[0]
    if K is None:
        K = n
    R = np.eye(n, dtype=np.float32)
    P = np.eye(n, dtype=np.float32)
    for _ in range(K):
        P = ((P @ A) > 0).astype(np.float32)
        R = ((R + P) > 0).astype(np.float32)
    return R.sum(1).astype(np.float32)


def blast_radius(A, K=None):
    """Normalised blast radius per node: fraction of the estate it can reach.

    ~1.0 in a flat estate (one cycle), ~1/k in a k-segment estate. Size-robust
    because it is a fraction, which is why a model fed this generalises to an
    unseen graph size.
    """
    n = A.shape[0]
    return (reachable_counts(A, K) / max(n, 1)).astype(np.float32)
