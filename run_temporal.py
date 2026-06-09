"""
Temporal demo: when time-ordering IS the signal.

    python run_temporal.py

Attack and benign streams share an identical static-graph distribution; they
differ only in whether the lateral-movement chain f -> ... -> g occurs in
time-respecting order. Any static model is therefore pinned at the base rate; a
temporal (time-respecting reachability) cover recovers the signal. Also reports
an early-detection (lead-time) metric a static classifier cannot even define.
Writes results/temporal_pr_auc.png and results/temporal_lead.png.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.temporal_data import make_dataset
from src.temporal_ops import temporal_features, static_features, lead_time
from src.temporal_models import stream_to_graph, make_mlp
from src.models import GCN
from src.plotstyle import use_style, BLUE, RED, REF

use_style()

EPOCHS, SEEDS = 60, (0, 1, 2)


def _oversample(X, y, rng):
    """Balance classes by oversampling the minority with replacement."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    take = rng.choice(pos, size=len(neg), replace=True)
    idx = rng.permutation(np.concatenate([neg, take]))
    return X[idx], y[idx]


def _mlp_ap(feat_fn, tr, ytr, te, yte, seed):
    rng = np.random.default_rng(seed)
    Xtr = np.array([feat_fn(s) for s in tr]); Xte = np.array([feat_fn(s) for s in te])
    Xb, yb = _oversample(Xtr, ytr, rng)
    clf = make_mlp(seed).fit(Xb, yb)
    return average_precision_score(yte, clf.predict_proba(Xte)[:, 1])


def _gcn_ap(tr, ytr, te, yte, seed):
    torch.manual_seed(seed)
    G = [stream_to_graph(s) for s in tr]; Gte = [stream_to_graph(s) for s in te]
    graphs = [{"Phat": torch.from_numpy(P), "x": torch.from_numpy(x)} for P, x in G]
    gte = [{"Phat": torch.from_numpy(P), "x": torch.from_numpy(x)} for P, x in Gte]
    model = GCN(in_dim=1, h=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.from_numpy(ytr.astype(np.float32))
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        logits = torch.stack([model(g).squeeze() for g in graphs])
        lossf(logits, yt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        s = torch.stack([model(g).squeeze() for g in gte]).sigmoid().numpy()
    return average_precision_score(yte, s)


def main():
    rows = {"static GCN": [], "MLP / static reach (v2)": [], "MLP / TEMPORAL cover": []}
    base = None
    for seed in SEEDS:
        tr, ytr = make_dataset(n=1000, seed=seed)
        te, yte = make_dataset(n=1000, seed=seed + 500)
        base = yte.mean()
        rows["static GCN"].append(_gcn_ap(tr, ytr, te, yte, seed))
        rows["MLP / static reach (v2)"].append(_mlp_ap(static_features, tr, ytr, te, yte, seed))
        rows["MLP / TEMPORAL cover"].append(_mlp_ap(temporal_features, tr, ytr, te, yte, seed))

    print(f"\nTemporal lateral-movement task  (base rate ~{base:.1%} positives, "
          f"identical static-graph distribution across classes)\n")
    print(f"{'model':28s} {'PR-AUC':>8s}")
    print("-" * 38)
    means = {k: float(np.mean(v)) for k, v in rows.items()}
    for k, v in means.items():
        print(f"{k:28s} {v:>8.3f}")
    print(f"\nbase rate: {base:.3f}\n")

    # lead-time distribution on true attacks (single seed)
    te, yte = make_dataset(n=1000, seed=999)
    leads = np.array([lead_time(s) for s, y in zip(te, yte) if y == 1])
    leads = leads[np.isfinite(leads)]
    detected_before = float((leads > 0).mean())
    print(f"attacks with a completed temporal path: {len(leads)}; "
          f"flagged BEFORE exfil: {detected_before:.0%}\n")

    _plot_prauc(means, base)
    _plot_lead(leads)
    print("Wrote results/temporal_pr_auc.png and results/temporal_lead.png")


def _plot_prauc(means, base):
    names = list(means); vals = [means[n] for n in names]
    colors = [RED, RED, BLUE]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(base, ls="--", lw=1, color=REF, label=f"base rate ({base:.2f})")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("Only the temporal cover recovers the ordering signal")
    ax.legend(loc="upper left")
    fig.savefig("results/temporal_pr_auc.png"); plt.close(fig)


def _plot_lead(leads):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(leads, bins=24, color=BLUE)
    ax.axvline(0, ls="--", lw=1.2, color=REF, label="exfil time")
    ax.set_xlabel("lead time before exfil (positive = detected early)")
    ax.set_ylabel("attacks")
    ax.set_title("Every detected attack is caught before exfiltration")
    ax.legend()
    fig.savefig("results/temporal_lead.png"); plt.close(fig)


if __name__ == "__main__":
    main()
