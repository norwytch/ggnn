"""
A Sheaf Neural Network baseline, the network realization (Hansen & Gebhart, *Sheaf
Neural Networks*, 2020) of the cellular sheaf Laplacian from the spectral theory of
Hansen & Ghrist, 2019.

A cellular sheaf puts a stalk R^d on each node and edge and a linear restriction map
on each node-edge incidence; the sheaf Laplacian L_F generalises the graph Laplacian,
and a sheaf-diffusion layer propagates stalk features through (I - L_F).

The catch this baseline is here to expose: on the tasks in this repo the node feature
is a constant and the graphs are vertex-transitive, so there is nothing to condition a
restriction map on. The sheaf is therefore homogeneous (one learned d x d map for every
incidence), and the sheaf Laplacian factors as L_graph (x) M. Diffusion then reduces to
ordinary graph diffusion mixed across the d stalk dimensions, i.e. a multi-channel GCN.
That is exactly why we expect it to sit at the base rate like GCN/GIN: the obstruction
is the missing global signal, not the model. We give it learnable maps (strictly more
capable than the fixed sheaf of the original) so the negative result is not a strawman.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _sym_norm(A):
    """Symmetric normalised adjacency with self-loops, D^-1/2 (A+I) D^-1/2."""
    n = A.shape[0]
    Ah = A + torch.eye(n, dtype=A.dtype)
    d = Ah.sum(1)
    dinv = d.clamp(min=1e-8).pow(-0.5)
    return dinv.unsqueeze(1) * Ah * dinv.unsqueeze(0)


class SheafNN(nn.Module):
    """Homogeneous sheaf-diffusion network. Stalk dimension `d`, `layers` diffusion
    steps, graph-level readout. Reads g['Asym'] (symmetric adjacency) and g['x'] (the
    constant node feature)."""
    def __init__(self, in_dim=1, d=3, h=16, layers=2):
        super().__init__()
        self.d, self.h = d, h
        self.lift = nn.Linear(in_dim, d * h)
        # restriction structure on stalks: M = B^T B is PSD, the homogeneous sheaf
        self.B = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
        self.mix = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.head = nn.Linear(h, 1)

    def forward(self, g):
        A, x = g["Asym"], g["x"]
        n = A.shape[0]
        P = _sym_norm(A)                                   # node propagator
        M = self.B @ self.B.t()                            # stalk restriction (PSD)
        X = self.lift(x).view(n, self.d, self.h)           # (nodes, stalks, channels)
        for W in self.mix:
            X = torch.einsum("mn,ndh->mdh", P, X)          # diffuse over the graph
            X = torch.einsum("de,neh->ndh", M, X)          # transport across stalks
            X = torch.relu(W(X))
        return self.head(X.mean(dim=(0, 1)))               # pool nodes + stalks


class SheafEdgeScorer(nn.Module):
    """Edge-level sheaf detector for the LANL probe: the same homogeneous sheaf
    diffusion, but with sparse propagation so it scales to the ~4,000-node per-window
    access graphs, and an edge head that scores each authentication from its endpoint
    embeddings. Trained end-to-end on red-team labels, so this is an actual learned
    sheaf NN, not a fixed operator.

    forward(P, X, ei): P is a sparse normalised adjacency (n x n), X the node features
    (n x in_dim), ei the 2 x E endpoint index of the edges to score. Returns E logits.
    """
    def __init__(self, in_dim=2, d=2, h=8, layers=2):
        super().__init__()
        self.d, self.h = d, h
        self.lift = nn.Linear(in_dim, d * h)
        self.B = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
        self.mix = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.edge = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def node_embed(self, P, X):
        n = X.shape[0]
        M = self.B @ self.B.t()
        H = self.lift(X).view(n, self.d, self.h)
        for W in self.mix:
            H = torch.sparse.mm(P, H.reshape(n, self.d * self.h)).view(n, self.d, self.h)
            H = torch.einsum("de,neh->ndh", M, H)
            H = torch.relu(W(H))
        return H.mean(dim=1)                               # (n, h), pool stalks

    def forward(self, P, X, ei):
        H = self.node_embed(P, X)
        return self.edge(torch.cat([H[ei[0]], H[ei[1]]], dim=-1)).squeeze(-1)


def sparse_norm_adj(A):
    """Sparse symmetric-normalised adjacency (with self-loops) from a dense 0/1
    directed adjacency `A` (numpy). Returns a coalesced torch sparse tensor."""
    import numpy as np
    n = A.shape[0]
    S = ((np.asarray(A) + np.asarray(A).T) > 0).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    deg = S.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    rows, cols = np.nonzero(S)
    vals = (dinv[rows] * dinv[cols]).astype(np.float32)
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n, n)).coalesce()
