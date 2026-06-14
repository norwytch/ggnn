"""
Smoke tests for the palimpsest calibration gate: the synthetic loader, the sparse
adjacency, and that all three node models forward-run and train one step on a small
heterophilic graph. No download.

    pytest -q tests/test_palimpsest.py
"""
import numpy as np
import torch

from palimpsest.data import synthetic_heterophily
from palimpsest.models import build_adj, edge_index, REGISTRY


def test_synthetic_loader_shapes():
    ds = synthetic_heterophily(n=120, n_classes=3)
    n = ds["x"].shape[0]
    assert ds["y"].shape == (n,)
    assert ds["edges"].ndim == 2 and ds["edges"].shape[1] == 2
    assert ds["edges"].max() < n                       # valid node indices
    assert ds["train"].shape[1] == n and ds["n_classes"] == 3


def test_models_forward_and_step():
    ds = synthetic_heterophily(n=150, d_feat=8, n_classes=4)
    n = ds["x"].shape[0]
    adj = build_adj(ds["edges"], n)
    ei = edge_index(ds["edges"])
    x, y = torch.from_numpy(ds["x"]), torch.from_numpy(ds["y"])
    tr = torch.from_numpy(ds["train"][0])
    for name, Model in REGISTRY.items():
        torch.manual_seed(0)
        m = Model(x.shape[1], 16, ds["n_classes"])
        out = m(x, adj, ei)
        assert out.shape == (n, ds["n_classes"]) and torch.isfinite(out).all(), name
        # one optimisation step decreases the train loss for this model
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        lossf = torch.nn.CrossEntropyLoss()
        l0 = lossf(m(x, adj, ei)[tr], y[tr]).item()
        for _ in range(20):
            opt.zero_grad(); lossf(m(x, adj, ei)[tr], y[tr]).backward(); opt.step()
        assert lossf(m(x, adj, ei)[tr], y[tr]).item() < l0, name
