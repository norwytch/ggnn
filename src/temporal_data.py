"""
Event-stream generator for the temporal demo.

Each sample is a stream of timestamped authentication events (u, v, t). Every
stream -- attack or benign -- contains the SAME edges: a fixed lateral-movement
chain f -> m1 -> ... -> g, the same random background traffic, and an exfil event
out of g. So the *static* multigraph distribution is identical across classes;
collapsing time away destroys the only signal.

The classes differ solely in the timestamps of the chain edges:

  * attack: chain edges occur in increasing time order, so a time-respecting path
    f ~> g exists and completes before the exfil event.
  * benign: the same chain edges occur in scrambled order, so no time-respecting
    traversal of the chain exists.

Background edges carry random timestamps in both classes -- which is what lets a
benign stream *coincidentally* form a time-respecting f ~> g path now and then.
That residual is real, and is why the temporal detector is strong but not perfect.
"""
from __future__ import annotations
import numpy as np

N_NODES = 24
CHAIN = [0, 1, 2, 3, 4, 5]       # f = 0, g = 5; a 5-edge chain so a scrambled
                                 # benign ordering is only coincidentally
                                 # time-respecting ~1/5! of the time
F, G = CHAIN[0], CHAIN[-1]
EXFIL = N_NODES                  # sink node for the exfiltration event
T_CHAIN = 1.0                    # chain/background window
T_GAP = 0.4                      # exfil happens within this gap after the window
N_BACKGROUND = 14                # background edges per stream (tunes FP rate)


def _background(rng):
    edges = []
    for _ in range(N_BACKGROUND):
        u, v = rng.integers(0, N_NODES, size=2)
        if u != v:
            edges.append((int(u), int(v), float(rng.uniform(0, T_CHAIN + T_GAP))))
    return edges


def _chain_events(attack, rng):
    times = rng.uniform(0, T_CHAIN, size=len(CHAIN) - 1)
    if attack:
        times = np.sort(times)                       # increasing -> time-respecting
    else:
        # scrambled: guarantee it is not increasing along the chain
        times = np.sort(times)[::-1].copy()
        rng.shuffle(times)
    return [(CHAIN[i], CHAIN[i + 1], float(times[i])) for i in range(len(CHAIN) - 1)]


def make_stream(attack, rng):
    events = _background(rng) + _chain_events(attack, rng)
    t_exfil = float(T_CHAIN + rng.uniform(0, T_GAP))
    events.append((G, EXFIL, t_exfil))
    return {"events": events, "t_exfil": t_exfil, "y": int(attack)}


def make_dataset(n=600, pos_rate=0.07, seed=0):
    rng = np.random.default_rng(seed)
    streams = [make_stream(rng.random() < pos_rate, rng) for _ in range(n)]
    y = np.array([s["y"] for s in streams], np.int64)
    return streams, y
