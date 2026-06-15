"""
palimpsest M1 — the calibration gate.

    python run_gate.py --synthetic                 # plumbing, no download
    python run_gate.py --dataset roman_empire      # the real gate (downloads ~few MB)

Trains GCN, GCN+skip, and a learned-restriction-map diagonal Sheaf NN on a Platonov
heterophily dataset over the official splits, and reports the leaderboard metric.

This is the gate, not the payoff. Two checks:
  * Standard GNNs should be strong on heterophily (Platonov's finding). If GCN+skip is
    weak here, the harness is wrong -- debug before going further.
  * The Sheaf NN should at least match a plain GCN. If it underperforms an untuned GCN,
    the sheaf implementation is wrong, not sheaves -- fix before trusting any comparison.

Nulls (pre-registered, per the proposal): the Sheaf NN does not clearly beat a tuned
GCN+skip. Reproducing that null first validates the harness.
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from palimpsest.data import load, synthetic_heterophily
from palimpsest.models import build_adj, edge_index, REGISTRY


def score(logits, y, mask, metric):
    if metric == "accuracy":
        return (logits[mask].argmax(1) == y[mask]).float().mean().item()
    prob = torch.softmax(logits[mask], 1)[:, 1].detach().numpy()
    return roc_auc_score(y[mask].numpy(), prob)


def train_eval(Model, ds, adj, ei, x, y, split, seed, epochs, hidden, layers, patience):
    torch.manual_seed(seed)
    tr = torch.from_numpy(ds["train"][split])
    va = torch.from_numpy(ds["val"][split])
    te = torch.from_numpy(ds["test"][split])
    model = Model(x.shape[1], hidden, ds["n_classes"], layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-4)
    lossf = torch.nn.CrossEntropyLoss()
    best_val, best_test, stale = -1.0, 0.0, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        lossf(model(x, adj, ei)[tr], y[tr]).backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(x, adj, ei)
            v, t = score(out, y, va, ds["metric"]), score(out, y, te, ds["metric"])
            if v > best_val:
                best_val, best_test, stale = v, t, 0
            else:
                stale += 1
                if stale > patience:                            # early stop
                    break
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roman_empire")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a fabricated heterophily graph (no download)")
    ap.add_argument("--splits", type=int, default=2, help="how many of the 10 splits")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--patience", type=int, default=40, help="early-stop patience (in checks)")
    args = ap.parse_args()

    if args.synthetic:
        print("[synthetic heterophily graph -- plumbing, not a benchmark]\n")
        ds = synthetic_heterophily()
    else:
        ds = load(args.dataset)
        print(f"\n{args.dataset}: {ds['x'].shape[0]:,} nodes, {ds['x'].shape[1]} features, "
              f"{ds['n_classes']} classes, metric={ds['metric']}")

    n = ds["x"].shape[0]
    adj = build_adj(ds["edges"], n)
    ei = edge_index(ds["edges"])
    x, y = torch.from_numpy(ds["x"]), torch.from_numpy(ds["y"])
    n_splits = min(args.splits, ds["train"].shape[0])

    print(f"\n{'model':14s} {ds['metric']:>10s} (mean +/- std over "
          f"{n_splits} split{'s' if n_splits > 1 else ''})")
    print("-" * 44)
    results = {}
    for name, Model in REGISTRY.items():
        vals = [train_eval(Model, ds, adj, ei, x, y, s, seed=s, epochs=args.epochs,
                           hidden=args.hidden, layers=args.layers, patience=args.patience)
                for s in range(n_splits)]
        results[name] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"{name:14s} {np.mean(vals):>10.3f}  +/- {np.std(vals):.3f}")

    gcn = results["GCN"][0]; sheaf = results["SheafNN"][0]
    best_std = max(results["GCN+skip"][0], results.get("SAGE", (0, 0))[0])
    best_std_name = "GCN+skip" if results["GCN+skip"][0] >= results.get("SAGE", (0, 0))[0] else "SAGE"
    skip_ok = best_std > gcn + 0.03                          # architecture tweaks must help
    sheaf_ok = sheaf >= gcn                                  # SNN must at least match plain GCN
    calibrated = skip_ok and sheaf_ok
    print("\ngate checks:")
    print(f"  standard GNN strong on heterophily?  best = {best_std:.3f} ({best_std_name}) "
          f"({'ok' if skip_ok else 'FLAG: barely beats plain GCN'})")
    print(f"  sheaf NN at least matches plain GCN?  SheafNN = {sheaf:.3f} vs GCN {gcn:.3f} "
          f"({'ok' if sheaf_ok else 'FLAG: sheaf < plain GCN -> sheaf impl/tuning, not sheaves'})")
    if not calibrated:
        print("  verdict: harness NOT yet calibrated -- M1 is not passed. (Gate doing its job.)\n")
    elif sheaf > best_std + 0.01:
        print(f"  verdict: sheaf ({sheaf:.3f}) beats the best standard GNN "
              f"({best_std:.3f}, {best_std_name}) -- investigate against the pre-registered null.\n")
    else:
        print(f"  verdict: sheaf ({sheaf:.3f}) does NOT beat the best standard GNN "
              f"({best_std:.3f}, {best_std_name}) -- the pre-registered null.\n")


if __name__ == "__main__":
    main()
