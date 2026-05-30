"""
Access-graph generator for the static lateral-movement task.

A graph is a directed access graph; node feature is a constant (every host looks
identical locally), which is what pins 1-WL. Two classes:

  * flat (positive, rare ~3%): one directed n-cycle. Every host reaches every
    other -- maximal blast radius.
  * segmented (negative): k disjoint directed cycles. Blast radius ~1/k.

The two classes are *locally identical*: in both, every node has in-degree 1,
out-degree 1 and the same feature. They differ only in a global property
(reachability / number of components).

The split is by graph size: training graphs use several sizes, the test set uses
one size never seen in training -- a generalisation probe that a size-robust
signal (the blast-radius *fraction*) passes and a size-sensitive trick does not.
"""
from __future__ import annotations
import numpy as np


def _cycle(nodes):
    """Directed cycle through `nodes` (a list of node indices)."""
    m = len(nodes)
    A = np.zeros((m, m), np.float32)
    local = {g: i for i, g in enumerate(nodes)}
    for i, g in enumerate(nodes):
        h = nodes[(i + 1) % m]
        A[local[g], local[h]] = 1.0
    return A


def _flat(n):
    return _cycle(list(range(n)))


def _segmented(n, k):
    """k disjoint directed cycles partitioning n nodes (k must divide n)."""
    seg = n // k
    A = np.zeros((n, n), np.float32)
    for s in range(k):
        block = list(range(s * seg, (s + 1) * seg))
        A[np.ix_(block, block)] = _cycle(block)
    return A


def _divisors(n, lo=2, hi=6):
    return [k for k in range(lo, hi + 1) if n % k == 0 and n // k >= 2]


def _make_one(n, positive, rng):
    if positive:
        return _flat(n)
    k = int(rng.choice(_divisors(n)))
    return _segmented(n, k)


def make_split(n_graphs, sizes, pos_rate, seed):
    """A list of (A, y) with the given sizes drawn uniformly and `pos_rate`
    positives. Node features are implicit (all-ones); models add them."""
    rng = np.random.default_rng(seed)
    As, ys = [], []
    for _ in range(n_graphs):
        n = int(rng.choice(sizes))
        y = int(rng.random() < pos_rate)
        As.append(_make_one(n, bool(y), rng))
        ys.append(y)
    return As, np.asarray(ys, np.int64)


def make_dataset(seed=0, n_train=400, n_test=400, pos_rate=0.03,
                 train_sizes=(12, 18, 24, 30), test_size=36):
    """Train on several sizes, test on an unseen size. Returns
    (train_As, train_y, test_As, test_y)."""
    tr_As, tr_y = make_split(n_train, train_sizes, pos_rate, seed)
    te_As, te_y = make_split(n_test, (test_size,), pos_rate, seed + 1000)
    return tr_As, tr_y, te_As, te_y
