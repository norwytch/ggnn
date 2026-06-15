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


class SAGE(nn.Module):
    """GraphSAGE-mean with residual connections and layer norm -- Platonov et al.'s
    strongest standard baseline on the corrected heterophily suite (the real ~0.85
    opponent the sheaf has to beat). Separates self and neighbour transforms, so it does
    not force a node toward its neighbours' labels the way a symmetric GCN does."""
    def __init__(self, in_dim, h, n_classes, layers=5, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, h)
        self.self_lin = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.neigh_lin = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.out = nn.Linear(h, n_classes)
        self.dropout = dropout

    def forward(self, x, adj, ei):
        n = x.shape[0]
        src, dst = ei[0], ei[1]
        deg = torch.zeros(n, dtype=x.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1.0)
        x = torch.relu(self.inp(x))
        for sl, nl, norm in zip(self.self_lin, self.neigh_lin, self.norms):
            mean = torch.zeros_like(x).index_add_(0, dst, x[src]) / deg.view(n, 1)
            h = norm(torch.relu(sl(x) + nl(mean)))
            x = x + F.dropout(h, self.dropout, self.training)   # residual
        return self.out(x)


class OrthSheafNN(nn.Module):
    """An O(2)-bundle sheaf network (the orthogonal-map / Bundle NN corner of the NSD
    family). Each node gets a learned 2D rotation R_v = R(theta_v) with theta_v a
    function of its features; a neighbour's stalk is transported by R_v^T R_u before it
    is aggregated. A learned relative rotation can turn a heterophilic neighbour's
    evidence out of the way instead of smoothing it in -- the sheaf-theoretic statement
    of "discount/invert this neighbour." Normalised aggregation, residual diffusion,
    sparse scatter. d=2 stalks, f channels."""
    def __init__(self, in_dim, h, n_classes, layers=4, dropout=0.3):
        super().__init__()
        self.f = max(h // 2, 1)                                  # channels of 2D stalks
        self.lift = nn.Linear(in_dim, 2 * self.f)
        self.angle = nn.Linear(in_dim, 1)                        # per-node rotation angle
        self.mix = nn.ModuleList(nn.Linear(2 * self.f, 2 * self.f) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(2 * self.f) for _ in range(layers))
        self.out = nn.Linear(2 * self.f, n_classes)
        self.dropout = dropout

    def forward(self, x, adj, ei):
        n = x.shape[0]
        src, dst = ei[0], ei[1]
        th = self.angle(x).squeeze(-1)
        c, s = torch.cos(th), torch.sin(th)
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (n,2,2)
        deg = torch.zeros(n, dtype=x.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1.0)
        invsqrt = deg.pow(-0.5)
        H = self.lift(x).view(n, self.f, 2)
        for W, norm in zip(self.mix, self.norms):
            RvtRu = torch.bmm(R[dst].transpose(1, 2), R[src])   # (E,2,2)
            trans = torch.einsum("eij,efj->efi", RvtRu, H[src]) # transport neighbour stalk
            w = (invsqrt[dst] * invsqrt[src]).view(-1, 1, 1)    # normalised
            agg = torch.zeros_like(H).index_add_(0, dst, w * trans)
            upd = norm(torch.relu(W(agg.reshape(n, 2 * self.f))))
            H = H + F.dropout(upd, self.dropout, self.training).view(n, self.f, 2)
        return self.out(H.reshape(n, 2 * self.f))


REGISTRY = {"GCN": GCN, "GCN+skip": GCNSkip, "SAGE": SAGE, "SheafNN": OrthSheafNN}
