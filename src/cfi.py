"""
Cai-Fuerer-Immerman (CFI) graph pairs: the canonical hard instances for the WL
hierarchy. Over a base graph G, the construction yields two non-isomorphic graphs that
k-WL cannot tell apart until k grows past a threshold set by G (roughly its
treewidth/connectivity). They are the standard way to manufacture a pair that defeats a
chosen WL level.

Gadget (the even-subset form): for each base vertex v of degree d, add a "middle"
vertex for every even-size subset S of v's incident edges (2^(d-1) of them). For each
base edge add two "edge" vertices, e0 and e1. Connect middle (v,S) to e1 if e in S, else
e0. The untwisted graph uses even subsets at every vertex; the twisted graph flips one
vertex to odd subsets. The parity is a global invariant, so on a 2-edge-connected base
the two graphs are non-isomorphic, yet low-dimensional WL cannot see the global flip.

We build the pair and certify its WL level empirically with src/wl.py rather than
relying on a treewidth formula.
"""
from __future__ import annotations
import itertools
import numpy as np


def _subsets(items):
    for r in range(len(items) + 1):
        yield from itertools.combinations(items, r)


def _build(n, edges, twist_vertex):
    inc = {v: [] for v in range(n)}
    for ei, (u, v) in enumerate(edges):
        inc[u].append(ei)
        inc[v].append(ei)

    nodes = []
    edge_node = {}
    for ei in range(len(edges)):
        for b in (0, 1):
            edge_node[(ei, b)] = len(nodes)
            nodes.append(("e", ei, b))

    mid_node = {}
    for v in range(n):
        parity = 1 if v == twist_vertex else 0
        for S in _subsets(inc[v]):
            if len(S) % 2 == parity:
                mid_node[(v, frozenset(S))] = len(nodes)
                nodes.append(("m", v, frozenset(S)))

    N = len(nodes)
    A = np.zeros((N, N), np.float32)
    for (v, S), mi in mid_node.items():
        for e in inc[v]:
            b = 1 if e in S else 0
            ej = edge_node[(e, b)]
            A[mi, ej] = A[ej, mi] = 1.0
    return A


def cfi_pair(n, edges):
    """Return (untwisted, twisted) adjacency matrices over base graph (n vertices,
    `edges` as (u,v) pairs). The base graph should be 2-edge-connected for the pair to
    be non-isomorphic."""
    return _build(n, edges, twist_vertex=None), _build(n, edges, twist_vertex=0)


# A few small 2-edge-connected base graphs to draw CFI pairs from.
def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cycle_graph(n):
    return [(i, (i + 1) % n) for i in range(n)]


def grid_2d(r, c):
    def idx(i, j):
        return i * c + j
    e = []
    for i in range(r):
        for j in range(c):
            if j + 1 < c:
                e.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < r:
                e.append((idx(i, j), idx(i + 1, j)))
    return e
