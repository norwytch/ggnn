"""
Cover-combining models for the SRG task.

Both models share an identical architecture: per-node features from a set of
cover blocks, learnable per-block gates (the model 'designs' its operator by
weighting covers), an MLP, and a sum-readout classifier. They differ ONLY in
which covers are in the basis:

    WalkCoverNet : {walk cover, reachability cover}            -> cannot separate
    SieveNet     : {walk cover, reachability cover, sieve}     -> separates

So any gap is attributable to adding the sieve cover to the basis -- the GGNN
"design via covers" thesis, made load-bearing.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CoverCombiner(nn.Module):
    def __init__(self, block_dims, hidden=32):
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(len(block_dims)))   # learnable per-cover weights
        self.proj = nn.ModuleList(nn.Linear(d, hidden) for d in block_dims)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.lin = nn.Linear(hidden, 2)

    def forward(self, blocks, batch, num):
        g = torch.softmax(self.gates, 0)
        h = sum(g[i] * self.proj[i](blocks[i]) for i in range(len(blocks)))
        h = self.mlp(h)
        out = torch.zeros(num, h.shape[1], device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, h)
        return self.lin(out)

    def gate_weights(self):
        return torch.softmax(self.gates, 0).detach().cpu().numpy()
