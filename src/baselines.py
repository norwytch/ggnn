"""
Per-edge feature extraction for the LANL task, split into two blocks so the
ablation is honest:

  * BASELINE features -- the things a practitioner already computes without any
    GNN: edge/credential novelty and simple degrees. Novelty ("has this
    src->dst, or this user-on-this-host, ever been seen before?") is a famously
    strong lateral-movement signal on LANL, because red-team movement is so often
    a credential used on a host it has never touched. This block is the control:
    the bar that cover features must clear to justify themselves.

  * COVER features -- the GGNN piece, restricted to be tractable on a real graph:
    a per-node within-K-hop reachable-set size on the *windowed* access graph
    (an ego-net-bounded reachability cover). This is the same blast-radius signal
    as the synthetic demo, but here it is NOT handed the answer -- a real auth
    graph is one big component, so global reachability is uninformative and only
    the local, bounded version carries anything.

`run_lanl` fits the SAME classifier on BASELINE vs BASELINE+COVER and asks whether
the cover block adds recall at a low false-positive rate. A null result is a real
result and is reported as such.
"""
from __future__ import annotations
import numpy as np

from .operators import reachable_counts


# ---- baseline (no-GNN) features ----------------------------------------------

class History:
    """Accumulates which edges and (user, dst) pairs have been seen, in temporal
    order, so novelty is always computed against the PAST only (no leakage)."""
    def __init__(self):
        self.edges = set()
        self.user_dst = set()
        self.in_deg = {}       # cumulative global in-degree per computer
        self.out_deg = {}

    def novelty(self, a):
        new_edge = (a.src_comp, a.dst_comp) not in self.edges
        new_cred = (a.src_user, a.dst_comp) not in self.user_dst
        return new_edge, new_cred

    def update(self, a):
        self.edges.add((a.src_comp, a.dst_comp))
        self.user_dst.add((a.src_user, a.dst_comp))
        self.out_deg[a.src_comp] = self.out_deg.get(a.src_comp, 0) + 1
        self.in_deg[a.dst_comp] = self.in_deg.get(a.dst_comp, 0) + 1


BASELINE_NAMES = [
    "novel_edge", "novel_cred", "auth_failed",
    "src_out_window", "dst_in_window",
    "log_src_out_global", "log_dst_in_global",
]


def baseline_features(window, history):
    """One baseline feature row per edge in `window`, using `history` (past only)
    for novelty and global degree. Does not mutate history -- the caller updates
    it after scoring, to keep train/test causality clean."""
    out_w, in_w = {}, {}
    for a in window.edges:
        out_w[a.src_comp] = out_w.get(a.src_comp, 0) + 1
        in_w[a.dst_comp] = in_w.get(a.dst_comp, 0) + 1
    rows = []
    for a in window.edges:
        ne, nc = history.novelty(a)
        rows.append([
            float(ne), float(nc), float(not a.success),
            float(out_w[a.src_comp]), float(in_w[a.dst_comp]),
            np.log1p(history.out_deg.get(a.src_comp, 0)),
            np.log1p(history.in_deg.get(a.dst_comp, 0)),
        ])
    return np.asarray(rows, np.float32)


# ---- cover (GGNN) features ----------------------------------------------------

COVER_NAMES = ["src_reach_K", "dst_reach_K", "dst_in_reach_K"]


def cover_features(window, K=3):
    """One cover feature row per edge: bounded reachability on the windowed graph.

    For edge src->dst we expose how far the source can already move within K hops
    (src_reach_K), how far the destination can move (dst_reach_K), and how many
    nodes can reach the destination within K hops (dst_in_reach_K, from the
    transpose). These are the ego-net-restricted reachability cover -- the only
    form of the cover that stays informative on a single-component real graph.
    """
    A = window.A
    fwd = reachable_counts(A, K)            # nodes each node can reach within K
    rev = reachable_counts(A.T, K)          # nodes that can reach each node
    idx = window.index
    rows = []
    for a in window.edges:
        s, d = idx[a.src_comp], idx[a.dst_comp]
        rows.append([fwd[s], fwd[d], rev[d]])
    return np.asarray(rows, np.float32)


def edge_labels(window, red):
    """Red-team label per edge in `window` (1 = labelled lateral movement)."""
    return np.asarray(
        [int(red(a.src_user, a.src_comp, a.dst_comp, a.time)) for a in window.edges],
        np.int64,
    )
