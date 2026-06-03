"""
LANL authentication-graph loader for the real-data sanity check.

Dataset: *Comprehensive, Multi-Source Cyber-Security Events* (A. D. Kent, LANL,
2015) -- 58 days of deidentified authentication events with a small set of
labelled red-team logons. Public; see the dataset page for terms and the
download (the file is large, ~12 GB gzipped for `auth.txt.gz`).

    https://csr.lanl.gov/data/cyber1/

Two files matter here:

  * auth.txt.gz   -- one event per line, comma-separated:
        time, src_user@domain, dst_user@domain, src_computer, dst_computer,
        auth_type, logon_type, auth_orientation, success/failure
  * redteam.txt   -- the labels, one per line:
        time, user@domain, src_computer, dst_computer
    ~749 records; these are the ground-truth lateral-movement authentications.
    Treat the labels as a *lower bound* -- the red team was not exhaustively
    annotated, so some unlabelled events may also be malicious. That makes
    precision pessimistic, which we say out loud at evaluation time.

This module does NOT download anything. It streams a slice you point it at, bins
events into time windows, builds a directed computer-to-computer access graph per
window, and emits edge-level examples labelled against redteam.txt.

For running the harness without the real download, `synthetic_slice` fabricates a
tiny event stream with planted lateral movement. It is a SMOKE TEST for the
plumbing, not evidence of anything -- the scientific claim only means something on
the real slice.
"""
from __future__ import annotations
import gzip
from dataclasses import dataclass

import numpy as np


# ---- raw event parsing --------------------------------------------------------

@dataclass(frozen=True)
class Auth:
    time: int
    src_user: str
    dst_user: str
    src_comp: str
    dst_comp: str
    success: bool


def _open(path):
    """Open plain or gzipped text transparently."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def parse_auth_line(line):
    """Parse one auth.txt line into an Auth, or None if it is malformed or not a
    computer-to-computer logon we care about (self-auths are dropped: lateral
    movement is host-to-host)."""
    f = line.rstrip("\n").split(",")
    if len(f) < 9:
        return None
    time, su, du, sc, dc = f[0], f[1], f[2], f[3], f[4]
    orientation, result = f[7], f[8]
    if orientation != "LogOn" or sc == dc:
        return None
    try:
        t = int(time)
    except ValueError:
        return None
    return Auth(t, su, du, sc, dc, result == "Success")


def iter_auth(path, t_start=None, t_end=None):
    """Stream Auth events in [t_start, t_end). Both bounds optional."""
    with _open(path) as fh:
        for line in fh:
            a = parse_auth_line(line)
            if a is None:
                continue
            if t_start is not None and a.time < t_start:
                continue
            if t_end is not None and a.time >= t_end:
                break          # auth.txt is time-sorted, so we can stop early
            yield a


# ---- labels -------------------------------------------------------------------

def load_redteam(path, tol=0):
    """Return a matcher: red(user, src_comp, dst_comp, time) -> bool.

    A red-team record is (time, user, src_comp, dst_comp). We key on the
    (user, src, dst) triple and, if `tol` > 0, also require the event time to fall
    within `tol` seconds of a labelled time for that triple -- guarding against a
    benign re-use of the same triple far from the attack window.
    """
    times = {}                 # (user, src, dst) -> sorted list of label times
    with _open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) < 4:
                continue
            t, user, sc, dc = int(f[0]), f[1], f[2], f[3]
            times.setdefault((user, sc, dc), []).append(t)
    for k in times:
        times[k].sort()

    def red(user, src_comp, dst_comp, time):
        ts = times.get((user, src_comp, dst_comp))
        if ts is None:
            return False
        if tol == 0:
            return True
        i = np.searchsorted(ts, time)
        for j in (i - 1, i):
            if 0 <= j < len(ts) and abs(ts[j] - time) <= tol:
                return True
        return False

    return red


# ---- windowing ----------------------------------------------------------------

@dataclass
class Window:
    t0: int                    # window start (inclusive)
    t1: int                    # window end (exclusive)
    edges: list                # list of Auth in this window
    nodes: list                # sorted unique computer ids present
    index: dict                # computer id -> row in the window adjacency
    A: np.ndarray              # directed 0/1 adjacency over `nodes`


def build_windows(events, window_seconds):
    """Bin a time-sorted Auth stream into fixed windows and build a directed
    computer access graph for each. Yields Window objects."""
    cur, t0 = [], None
    for a in events:
        if t0 is None:
            t0 = a.time - (a.time % window_seconds)
        if a.time >= t0 + window_seconds:
            if cur:
                yield _make_window(cur, t0, t0 + window_seconds)
            # advance to the window containing this event (handles gaps)
            t0 = a.time - (a.time % window_seconds)
            cur = []
        cur.append(a)
    if cur:
        yield _make_window(cur, t0, t0 + window_seconds)


def _make_window(edges, t0, t1):
    nodes = sorted({a.src_comp for a in edges} | {a.dst_comp for a in edges})
    index = {c: i for i, c in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), np.float32)
    for a in edges:
        A[index[a.src_comp], index[a.dst_comp]] = 1.0
    return Window(t0, t1, edges, nodes, index, A)


# ---- synthetic smoke-test slice ----------------------------------------------

def synthetic_slice(seed=0, n_comp=120, n_windows=40, window_seconds=3600):
    """Fabricate a tiny event stream with planted lateral movement, for exercising
    the harness end to end WITHOUT the real download. Not a benchmark.

    Benign traffic is a sparse stable set of src->dst pairs that recur. Attacks are
    chains of *novel* edges that also extend a node's reach -- so a good detector
    should beat chance, but nothing here says covers help on real data.
    """
    rng = np.random.default_rng(seed)
    comps = [f"C{i}" for i in range(n_comp)]
    # a stable benign edge set that recurs every window
    benign = set()
    while len(benign) < n_comp * 3:
        s, d = rng.integers(0, n_comp, 2)
        if s != d:
            benign.add((comps[s], comps[d]))
    benign = sorted(benign)

    events, labels = [], []
    for w in range(n_windows):
        base_t = w * window_seconds
        # recurring benign traffic (a noisy subsample each window)
        for (s, d) in benign:
            if rng.random() < 0.6:
                t = base_t + int(rng.integers(0, window_seconds))
                events.append((t, "U_bg", s, d)); labels.append(0)
        # a red-team chain in some windows: a path of fresh edges
        if w >= n_windows // 3 and rng.random() < 0.5:
            start = int(rng.integers(0, n_comp))
            hop = start
            for _ in range(int(rng.integers(2, 5))):
                nxt = int(rng.integers(0, n_comp))
                if nxt == hop:
                    continue
                t = base_t + int(rng.integers(0, window_seconds))
                events.append((t, "U_red", comps[hop], comps[nxt])); labels.append(1)
                hop = nxt
    order = np.argsort([e[0] for e in events], kind="stable")
    events = [events[i] for i in order]
    labels = [labels[i] for i in order]

    auths = [Auth(t, u, u, s, d, True) for (t, u, s, d) in events]
    is_red = np.asarray(labels, np.int64)

    def red(user, src_comp, dst_comp, time):
        # exact-triple match against planted attack edges (built once below)
        return (user, src_comp, dst_comp) in _planted

    _planted = {(u, s, d) for (t, u, s, d), y in zip(events, labels) if y == 1}
    return auths, red, window_seconds
