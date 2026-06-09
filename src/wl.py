"""
A small Weisfeiler-Leman oracle, so the repo certifies the WL level of a graph pair
instead of citing it.

We implement the folklore variant (k-FWL), which is the one with the clean update and
the standard equivalence k-FWL == (k+1)-WL in distinguishing power:

  * 1-WL          -- ordinary colour refinement on vertices.
  * 2-FWL (= 3-WL) -- refinement on ordered pairs.
  * 3-FWL (= 4-WL) -- refinement on ordered triples.

To make colours comparable across two graphs, every refinement round relabels the
signatures of all graphs through one shared map, which is the usual "run WL on the
disjoint union" trick. Two graphs are distinguished at a given level iff their final
colour histograms differ.

Graphs are tiny here (16 nodes), so the naive k-FWL loop (n^k tuples, n-way
aggregation) is fine: 3-FWL on 16 nodes is ~65k inner steps per round.
"""
from __future__ import annotations
import itertools
import numpy as np


def _init_color(A, tup):
    """Atomic type of an ordered k-tuple: the equality pattern among its indices plus
    the edge bits between every pair. This is the ordered isomorphism type of the
    induced (multi-)subgraph, the standard k-FWL initial colouring."""
    k = len(tup)
    eq = tuple((tup[i] == tup[j]) for i in range(k) for j in range(i + 1, k))
    edges = tuple(int(A[tup[i], tup[j]]) for i in range(k) for j in range(i + 1, k))
    return (eq, edges)


def _relabel(per_graph_sigs):
    """Map signatures to integer colour ids through one shared dictionary, so colours
    mean the same thing across graphs. Returns (list of {tuple: id}, n_colours)."""
    code = {}
    out = []
    for sigs in per_graph_sigs:
        col = {}
        for tup, s in sigs.items():
            if s not in code:
                code[s] = len(code)
            col[tup] = code[s]
        out.append(col)
    return out, len(code)


def _histogram(col):
    """Canonical colour histogram of one graph: sorted multiset of colour counts."""
    counts = {}
    for c in col.values():
        counts[c] = counts.get(c, 0) + 1
    return tuple(sorted(counts.values()))


def fwl_histograms(adjs, k, max_rounds=None):
    """Run k-FWL on each adjacency matrix in `adjs` with shared relabelling, to a
    stable colouring, and return one canonical colour histogram per graph."""
    adjs = [np.asarray(A) for A in adjs]
    tuples = [list(itertools.product(range(A.shape[0]), repeat=k)) for A in adjs]
    sigs = [{t: _init_color(A, t) for t in ts} for A, ts in zip(adjs, tuples)]
    cols, n_col = _relabel(sigs)

    rounds = max_rounds if max_rounds is not None else max(A.shape[0] for A in adjs) ** k
    for _ in range(rounds):
        sigs = []
        for A, ts, col in zip(adjs, tuples, cols):
            n = A.shape[0]
            g = {}
            for t in ts:
                neigh = []
                for x in range(n):
                    neigh.append(tuple(col[t[:i] + (x,) + t[i + 1:]] for i in range(k)))
                neigh.sort()
                g[t] = (col[t], tuple(neigh))
            sigs.append(g)
        new_cols, new_n = _relabel(sigs)
        cols = new_cols
        if new_n == n_col:                      # colour partition stabilised
            break
        n_col = new_n
    return [_histogram(c) for c in cols]


def wl1_histograms(adjs, max_rounds=None):
    """Ordinary 1-WL vertex colour refinement, shared relabelling, per-graph
    histograms. Separate from k-FWL because it refines vertices, not tuples."""
    adjs = [(np.asarray(A) > 0).astype(int) for A in adjs]
    sigs = [{v: (0,) for v in range(A.shape[0])} for A in adjs]   # uniform start
    cols, n_col = _relabel(sigs)
    rounds = max_rounds if max_rounds is not None else max(A.shape[0] for A in adjs)
    for _ in range(rounds):
        sigs = []
        for A, col in zip(adjs, cols):
            n = A.shape[0]
            g = {}
            for v in range(n):
                nb = sorted(col[u] for u in range(n) if A[v, u])
                g[v] = (col[v], tuple(nb))
            sigs.append(g)
        new_cols, new_n = _relabel(sigs)
        cols = new_cols
        if new_n == n_col:
            break
        n_col = new_n
    return [_histogram(c) for c in cols]


def distinguishes(A, B, level):
    """True if the WL test at `level` tells A and B apart.

    level: "1-WL", "3-WL" (= 2-FWL), or "4-WL" (= 3-FWL).
    """
    if level == "1-WL":
        ha, hb = wl1_histograms([A, B])
    elif level in ("3-WL", "2-FWL"):
        ha, hb = fwl_histograms([A, B], k=2)
    elif level in ("4-WL", "3-FWL"):
        ha, hb = fwl_histograms([A, B], k=3)
    else:
        raise ValueError(f"unknown level {level!r}")
    return ha != hb
