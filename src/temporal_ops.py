"""
Temporal and static reachability operators, and the feature blocks built on them.

The temporal cover is *time-respecting reachability*: the set of nodes reachable
from a source along a path whose edge timestamps strictly increase. It is computed
by a single time-ordered sweep -- the temporal analogue of summing powers of the
adjacency, with event order replacing multiplication order. Categorically, the
set reachable into a node is its causal past.

The static reachability operator (v2) is the time-agnostic version: ordinary
graph reachability on the collapsed edge set. Because the chain edges are always
present statically, g is *always* statically reachable from f in both classes --
so this feature carries no class signal, by construction.
"""
from __future__ import annotations
import numpy as np
from .temporal_data import N_NODES, F, G, CHAIN


def temporal_arrivals(events, src=F):
    """Earliest time-respecting arrival time per node, reachable from `src`.

    One ordered sweep: an event (u, v, t) extends reachability to v iff u is
    already reachable by time t. Returns {node: earliest_arrival_time}.
    """
    arrival = {src: -np.inf}                      # source available from the start
    for u, v, t in sorted(events, key=lambda e: e[2]):
        au = arrival.get(u)
        if au is not None and au <= t and (v not in arrival or t < arrival[v]):
            arrival[v] = t
    return arrival


def static_reachable(events, src=F):
    """Time-agnostic reachable set from `src` on the collapsed edge set."""
    adj = {}
    for u, v, _ in events:
        adj.setdefault(u, set()).add(v)
    seen, stack = {src}, [src]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return seen


def temporal_features(stream):
    """Temporal cover feature block for one stream."""
    arr = temporal_arrivals(stream["events"])
    g_time = arr.get(G, np.inf)
    g_before_exfil = float(g_time < stream["t_exfil"])
    chain_frac = np.mean([c in arr for c in CHAIN])
    reach_frac = len(arr) / (N_NODES + 1)
    arr_norm = 1.0 if not np.isfinite(g_time) else float(np.clip(g_time, 0, 1))
    return np.array([g_before_exfil, chain_frac, reach_frac, arr_norm], np.float32)


def static_features(stream):
    """Static reachability feature block (v2) for one stream."""
    reach = static_reachable(stream["events"])
    g_reach = float(G in reach)
    chain_frac = np.mean([c in reach for c in CHAIN])
    reach_frac = len(reach) / (N_NODES + 1)
    return np.array([g_reach, chain_frac, reach_frac], np.float32)


def lead_time(stream):
    """Exfil time minus temporal arrival at g (lead time of detection).

    Positive => the time-respecting path completed before data left. np.nan if g
    is never temporally reachable.
    """
    arr = temporal_arrivals(stream["events"])
    g_time = arr.get(G, np.inf)
    if not np.isfinite(g_time):
        return np.nan
    return float(stream["t_exfil"] - g_time)
