"""
Models for the temporal demo.

Three readers of the same event streams:

  * a static GCN over the collapsed access graph (node feature constant) -- the
    standard-model baseline;
  * an MLP over static reachability features (v2);
  * an MLP over the temporal cover features.

The first two see only the static graph, whose distribution is identical across
classes, so both are pinned at the base rate. Only the temporal MLP reads the
ordering that separates attack from benign.
"""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPClassifier

from .temporal_data import N_NODES, EXFIL
from .operators import gcn_norm


def stream_to_graph(stream):
    """Collapse a stream to a static symmetric access graph for the GCN."""
    n = N_NODES + 1                              # include the exfil sink
    A = np.zeros((n, n), np.float32)
    for u, v, _ in stream["events"]:
        A[u, v] = 1.0
    return gcn_norm(A), np.ones((n, 1), np.float32)


def make_mlp(seed=0):
    return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                         alpha=1e-3, random_state=seed)
