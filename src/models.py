"""
Models for the static demo, sharing one readout head so the only thing that
varies is what each can *see*:

  * GCN, GIN -- 1-WL-bounded message passing on the constant node feature. Both
    classes are locally identical, so these produce the same graph embedding for
    flat and segmented and land at the base rate.
  * GCNPlus  -- the same GCN, but with the per-node reachability (blast-radius)
    feature concatenated to its input. Once the global signal is handed to it as
    a node feature, ordinary message passing solves the task.
  * CoverNet -- an MLP over the bounded-reachability *cover* features (walks
    arriving, closed walks, blast radius), then the shared readout head.

Each forward() takes a per-graph dict of precomputed tensors (see
run_experiment.collate) and returns a single graph-level logit.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _Head(nn.Module):
    """Shared graph-level classifier: a linear map from a pooled embedding."""
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(h, 1)

    def forward(self, pooled):
        return self.lin(pooled)


class GCN(nn.Module):
    def __init__(self, in_dim=1, h=16, key="x"):
        super().__init__()
        self.key = key
        self.l1 = nn.Linear(in_dim, h)
        self.l2 = nn.Linear(h, h)
        self.head = _Head(h)

    def forward(self, g):
        P, x = g["Phat"], g[self.key]
        H = torch.relu(P @ self.l1(x))
        H = torch.relu(P @ self.l2(H))
        return self.head(H.mean(0))


class GIN(nn.Module):
    def __init__(self, in_dim=1, h=16):
        super().__init__()
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.eps2 = nn.Parameter(torch.zeros(1))
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.mlp2 = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.head = _Head(h)

    def forward(self, g):
        A, x = g["Asym"], g["x"]
        H = self.mlp1((1 + self.eps1) * x + A @ x)
        H = self.mlp2((1 + self.eps2) * H + A @ H)
        return self.head(H.sum(0))            # sum readout, the GIN default


class GCNPlus(GCN):
    """GCN given the reachability feature: same class, just reads the augmented
    node input `xplus` instead of the constant `x`."""
    def __init__(self, in_dim, h=16):
        super().__init__(in_dim=in_dim, h=h, key="xplus")


class CoverNet(nn.Module):
    """MLP over the per-node cover-feature block, then the shared readout head."""
    def __init__(self, in_dim, h=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU())
        self.head = _Head(h)

    def forward(self, g):
        H = self.mlp(g["cover"])
        return self.head(H.mean(0))
