"""
Node-classification model matrix for the palimpsest calibration gate.

  * GCN        -- a plain 2-layer sparse GCN; the 1-WL floor.
  * GCNSkip    -- GCN with residual connections and layer norm; Platonov et al.'s
                  "a tuned standard GNN is hard to beat on heterophily" baseline.
  * DiagSheafNN -- learned-restriction-map sheaf diffusion (the Neural Sheaf Diffusion
                  family, Bodnar et al. 2022), diagonal maps. Each node's restriction
                  map is learned FROM its features, which is exactly what ggnn's
                  featureless tasks denied it; on a heterophilic edge a learned sign flip
                  can invert/discount a neighbour's evidence.

All share forward(x, adj, edge_index); GCN/GCNSkip use the sparse normalised adjacency,
the sheaf model uses the edge index (sparse scatter, scales to ~20k nodes).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adj(edges, n):
    """Sparse symmetric-normalised adjacency with self-loops, from a (E, 2) edge list."""
    src, dst = edges[:, 0], edges[:, 1]
    loops = np.arange(n)
    src = np.concatenate([src, loops]); dst = np.concatenate([dst, loops])
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    vals = (dinv[dst] * dinv[src]).astype(np.float32)
    idx = torch.from_numpy(np.stack([dst, src]))               # rows = message target
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n, n)).coalesce()


def edge_index(edges):
    return torch.from_numpy(edges.T.copy())                     # (2, E)


class GCN(nn.Module):
    def __init__(self, in_dim, h, n_classes, layers=2, dropout=0.5):
        super().__init__()
        self.lins = nn.ModuleList(
            nn.Linear(in_dim if i == 0 else h, h) for i in range(layers))
        self.out = nn.Linear(h, n_classes)
        self.dropout = dropout

    def forward(self, x, adj, ei=None):
        for lin in self.lins:
            x = F.dropout(torch.relu(torch.sparse.mm(adj, lin(x))), self.dropout, self.training)
        return self.out(x)


class GCNSkip(nn.Module):
    def __init__(self, in_dim, h, n_classes, layers=3, dropout=0.5):
        super().__init__()
        self.inp = nn.Linear(in_dim, h)
        self.lins = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.out = nn.Linear(h, n_classes)
        self.dropout = dropout

    def forward(self, x, adj, ei=None):
        x = torch.relu(self.inp(x))
        for lin, norm in zip(self.lins, self.norms):
            h = norm(torch.relu(torch.sparse.mm(adj, lin(x))))
            x = x + F.dropout(h, self.dropout, self.training)    # residual
        return self.out(x)


class DiagSheafNN(nn.Module):
    def __init__(self, in_dim, h, n_classes, d=4, layers=2, dropout=0.5):
        super().__init__()
        self.d, self.h = d, h
        self.lift = nn.Linear(in_dim, d * h)
        self.phi = nn.Linear(in_dim, d)                          # per-node diagonal map
        self.mix = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.out = nn.Linear(h, n_classes)
        self.dropout = dropout

    def forward(self, x, adj, ei):
        n = x.shape[0]
        src, dst = ei[0], ei[1]
        a = torch.tanh(self.phi(x))                              # (n, d), sign = reflect
        deg = torch.zeros(n, dtype=x.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1.0)
        H = self.lift(x).view(n, self.d, self.h)
        for W in self.mix:
            msg = (a[dst] * a[src]).unsqueeze(-1) * H[src]       # transported stalk (E,d,h)
            agg = torch.zeros_like(H).index_add_(0, dst, msg) / deg.view(n, 1, 1)
            H = F.dropout(torch.relu(W(agg)), self.dropout, self.training)
        return self.out(H.mean(1))                              # pool stalks


REGISTRY = {"GCN": GCN, "GCN+skip": GCNSkip, "DiagSheafNN": DiagSheafNN}
