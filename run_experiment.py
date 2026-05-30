"""
Static demo: 1-WL GNNs vs the global-reachability signal.

    python run_experiment.py

Classifies flat (one directed cycle) vs segmented (k disjoint cycles) access
graphs under realistic class imbalance (~3% positives), trained on several graph
sizes and tested on an unseen size. Reports PR-AUC and recall at a fixed 1% FPR
over 3 seeds, plus a trivial LogReg-on-#components baseline, and writes
results/pr_auc.png and results/separation.png.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve

from src.data import make_dataset
from src.operators import gcn_norm, n_components, blast_radius
from src.covers import walk_cover, reachability_cover
from src.models import GCN, GIN, GCNPlus, CoverNet

K_WALK, HIDDEN, EPOCHS, SEEDS = 4, 16, 80, (0, 1, 2)


def recall_at_fpr(y, score, fpr_target=0.01):
    """Recall (TPR) at the largest threshold whose FPR <= fpr_target."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def collate(A):
    """Per-graph tensors the models read."""
    n = A.shape[0]
    ones = np.ones((n, 1), np.float32)
    cover = np.concatenate([walk_cover(A, K_WALK), reachability_cover(A, n)], 1)
    return {
        "Phat": torch.from_numpy(gcn_norm(A)),
        "Asym": torch.from_numpy(((A + A.T) > 0).astype(np.float32)),
        "x": torch.from_numpy(ones),
        "xplus": torch.from_numpy(np.concatenate([ones, blast_radius(A)[:, None]], 1)),
        "cover": torch.from_numpy(cover.astype(np.float32)),
    }


def train_eval(ModelCls, kwargs, tr, te, ytr, yte, seed):
    torch.manual_seed(seed)
    model = ModelCls(**kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.from_numpy(ytr.astype(np.float32))
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        logits = torch.stack([model(g).squeeze() for g in tr])
        lossf(logits, yt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        score = torch.stack([model(g).squeeze() for g in te]).sigmoid().numpy()
    return average_precision_score(yte, score), recall_at_fpr(yte, score)


def main():
    base_pr, recalls, prs = None, {}, {}
    cfgs = {
        "GCN (1-WL)":            (GCN,      {"in_dim": 1, "h": HIDDEN}),
        "GIN (1-WL)":            (GIN,      {"in_dim": 1, "h": HIDDEN}),
        "GCNPlus (+ reach)":     (GCNPlus,  {"in_dim": 2, "h": HIDDEN}),
        "CoverNet":              (CoverNet, {"in_dim": None, "h": HIDDEN}),
    }

    for name in cfgs:
        prs[name], recalls[name] = [], []
    lr_pr, lr_rc = [], []

    for seed in SEEDS:
        tr_As, ytr, te_As, yte = make_dataset(seed=seed)
        base_pr = yte.mean()
        tr = [collate(A) for A in tr_As]
        te = [collate(A) for A in te_As]
        cover_dim = tr[0]["cover"].shape[1]

        # trivial baseline: LogReg on the number of connected components
        ftr = np.array([[n_components(A)] for A in tr_As], np.float32)
        fte = np.array([[n_components(A)] for A in te_As], np.float32)
        lr = LogisticRegression(class_weight="balanced", max_iter=500).fit(ftr, ytr)
        s = lr.predict_proba(fte)[:, 1]
        lr_pr.append(average_precision_score(yte, s)); lr_rc.append(recall_at_fpr(yte, s))

        for name, (Cls, kw) in cfgs.items():
            kw = dict(kw)
            if name == "CoverNet":
                kw["in_dim"] = cover_dim
            pr, rc = train_eval(Cls, kw, tr, te, ytr, yte, seed)
            prs[name].append(pr); recalls[name].append(rc)

    rows = [("LogReg / #components", np.mean(lr_pr), np.mean(lr_rc))]
    rows += [(n, np.mean(prs[n]), np.mean(recalls[n])) for n in cfgs]

    print(f"\nStatic lateral-movement task  (base rate ~{base_pr:.1%} positives, "
          f"tested on an UNSEEN graph size)\n")
    print(f"{'model':24s} {'PR-AUC':>8s} {'recall@1%FPR':>14s}")
    print("-" * 48)
    for name, pr, rc in rows:
        print(f"{name:24s} {pr:>8.3f} {rc:>14.2f}")
    print(f"\nbase rate (PR-AUC of a random ranker): {base_pr:.3f}\n")

    _plot_prauc({n: pr for n, pr, _ in rows}, base_pr)
    _plot_separation()
    print("Wrote results/pr_auc.png and results/separation.png")


def _plot_prauc(prauc, base):
    names = list(prauc); vals = [prauc[n] for n in names]
    blind = {"GCN (1-WL)", "GIN (1-WL)"}
    colors = ["#d93025" if n in blind else "#1a73e8" for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(base, ls="--", lw=1, color="#5f6368", label=f"base rate ({base:.2f})")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("Flat vs segmented under class imbalance, unseen size:\n"
                 "1-WL GNNs sit at the base rate; the reachability signal solves it")
    ax.legend(loc="center right"); fig.tight_layout()
    fig.savefig("results/pr_auc.png", dpi=140); plt.close(fig)


def _plot_separation():
    # blast radius per node for a representative flat vs segmented graph
    from src.data import _flat, _segmented
    flat = np.concatenate([blast_radius(_flat(36))])
    seg = np.concatenate([blast_radius(_segmented(36, 3))])
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bins = np.linspace(0, 1.02, 30)
    ax.hist(seg, bins=bins, color="#1a73e8", alpha=0.8, label="segmented (k=3): ~1/k")
    ax.hist(flat, bins=bins, color="#d93025", alpha=0.8, label="flat: ~1.0")
    ax.set_xlabel("normalised blast radius (fraction of estate reachable)")
    ax.set_ylabel("nodes")
    ax.set_title("The separating signal is global, not local:\n"
                 "node blast radius is bimodal by class but invisible to 1-WL")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/separation.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
