"""
Real-data sanity check: do cover features earn their keep on LANL?

    python run_lanl.py --smoke                       # synthetic plumbing test
    python run_lanl.py --auth PATH --redteam PATH     # the real slice

This is the experiment the synthetic demos cannot be: a task where the separating
signal is NOT handed to the model by construction. A real authentication graph is
one giant connected component, so "count components / global reachability" -- the
trick that scores 1.0 on the synthetic task -- is useless here. The question is
whether a bounded, ego-net-restricted reachability *cover* adds anything over the
strong no-GNN baseline (edge/credential novelty + degrees).

Method, kept deliberately austere so the comparison is clean:
  * Stream a slice, bin into time windows, build a per-window access graph.
  * TEMPORAL split: earliest `--train-frac` of windows train, the rest test.
    Never a random split -- that leaks the future. Novelty is always computed
    against past windows only.
  * Fit the SAME classifier (logistic regression, balanced) twice: on BASELINE
    features, then on BASELINE + COVER features. If the cover block does not move
    recall at a low FPR, we say so. A null result is the honest result.

Metrics are imbalance-aware: PR-AUC, recall at a fixed low FPR, and alerts/day at
that operating point (what an analyst actually pays for). Red-team labels are a
lower bound, so reported precision is pessimistic; we print that caveat.
"""
from __future__ import annotations
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from src.lanl import iter_auth, load_redteam, build_windows, synthetic_slice
from src.baselines import (
    History, baseline_features, cover_features, edge_labels,
    BASELINE_NAMES, COVER_NAMES,
)

FPR_TARGET = 0.001          # 0.1% -- a realistic alerting budget, not 1%
COVER_K = 3


def recall_at_fpr(y, score, fpr_target=FPR_TARGET):
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def alerts_per_day(y, score, n_days, fpr_target=FPR_TARGET):
    """Number of positives raised per day at the threshold meeting `fpr_target`,
    a practitioner-facing read on the false-positive load."""
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where(fpr <= fpr_target)[0]
    if len(ok) == 0:
        return float("nan")
    t = thr[ok[-1]]
    return float((score >= t).sum()) / max(n_days, 1)


def assemble(windows, red, K=COVER_K):
    """Walk windows in time order, building feature blocks and labels. Novelty
    uses only past windows: features are computed against `history` BEFORE the
    window is folded in. Returns (Xbase, Xcover, y) and the window count."""
    history = History()
    Xb, Xc, ys = [], [], []
    for w in windows:
        Xb.append(baseline_features(w, history))
        Xc.append(cover_features(w, K))
        ys.append(edge_labels(w, red))
        for a in w.edges:               # only now does this window become "past"
            history.update(a)
    return (np.concatenate(Xb), np.concatenate(Xc), np.concatenate(ys),
            len(Xb))


def fit_score(Xtr, ytr, Xte):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def temporal_split_windows(windows, train_frac):
    """Split the LIST of windows by time, then assemble each side. Splitting
    windows (not rows) keeps a window's edges together and the future unseen."""
    k = int(len(windows) * train_frac)
    return windows[:k], windows[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run on a fabricated slice (plumbing test, not a benchmark)")
    ap.add_argument("--auth", help="path to auth.txt[.gz]")
    ap.add_argument("--redteam", help="path to redteam.txt[.gz]")
    ap.add_argument("--t-start", type=int, default=None)
    ap.add_argument("--t-end", type=int, default=None)
    ap.add_argument("--window", type=int, default=3600, help="window length (s)")
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()

    if args.smoke:
        print("[smoke test: synthetic slice -- plumbing only, not evidence]\n")
        auths, red, window_seconds = synthetic_slice(window_seconds=args.window)
        windows = list(build_windows(iter(auths), window_seconds))
    else:
        if not (args.auth and args.redteam):
            ap.error("provide --auth and --redteam, or use --smoke")
        red = load_redteam(args.redteam)
        events = iter_auth(args.auth, args.t_start, args.t_end)
        windows = list(build_windows(events, args.window))

    tr_w, te_w = temporal_split_windows(windows, args.train_frac)
    Xb_tr, Xc_tr, ytr, _ = assemble(tr_w, red)
    Xb_te, Xc_te, yte, n_te_win = assemble(te_w, red)
    n_days = max(1, (n_te_win * args.window) / 86400)

    if yte.sum() == 0 or ytr.sum() == 0:
        print("No red-team positives in a split -- widen the slice "
              "(--t-start/--t-end) to include labelled attack windows.")
        return

    runs = {
        "baseline (novelty+degree)": (Xb_tr, Xb_te),
        "baseline + cover": (np.hstack([Xb_tr, Xc_tr]), np.hstack([Xb_te, Xc_te])),
    }
    results = {}
    for name, (Xtr, Xte) in runs.items():
        s = fit_score(Xtr, ytr, Xte)
        results[name] = {
            "pr_auc": average_precision_score(yte, s),
            "recall": recall_at_fpr(yte, s),
            "alerts": alerts_per_day(yte, s, n_days),
            "score": s,
        }

    base_rate = yte.mean()
    print(f"\nLANL edge-level lateral-movement task")
    print(f"  test edges: {len(yte):,}   positives: {int(yte.sum())} "
          f"({base_rate:.3%})   test span: ~{n_days:.1f} days\n")
    print(f"{'model':28s} {'PR-AUC':>8s} {f'recall@{FPR_TARGET:.1%}FPR':>16s} "
          f"{'alerts/day':>12s}")
    print("-" * 68)
    for name, r in results.items():
        print(f"{name:28s} {r['pr_auc']:>8.3f} {r['recall']:>16.2f} "
              f"{r['alerts']:>12.1f}")

    b, c = results["baseline (novelty+degree)"], results["baseline + cover"]
    dpr, drc = c["pr_auc"] - b["pr_auc"], c["recall"] - b["recall"]
    print(f"\ncover delta: PR-AUC {dpr:+.3f}, recall@{FPR_TARGET:.1%}FPR {drc:+.2f}")
    verdict = ("covers ADD signal over the baseline" if (dpr > 0.01 or drc > 0.02)
               else "covers do NOT clear the baseline here -- honest null")
    print(f"verdict: {verdict}")
    print("\nNote: red-team labels are a lower bound; some 'false positives' may be "
          "unlabelled malicious, so precision/alerts are pessimistic.\n")

    _plot_pr(yte, results, base_rate)
    print("Wrote results/lanl_pr.png")


def _plot_pr(y, results, base_rate):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name, r in results.items():
        p, rc, _ = precision_recall_curve(y, r["score"])
        ax.plot(rc, p, lw=2, label=f"{name} (AP={r['pr_auc']:.3f})")
    ax.axhline(base_rate, ls="--", lw=1, color="#5f6368",
               label=f"base rate ({base_rate:.3%})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("LANL lateral movement: does the cover block beat novelty+degree?")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig("results/lanl_pr.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
