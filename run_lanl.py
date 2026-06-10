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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.lanl import iter_auth, load_redteam, build_windows, synthetic_slice
from src.baselines import (
    History, baseline_features, cover_features, edge_labels,
    BASELINE_NAMES, COVER_NAMES,
)
from src.plotstyle import use_style, BLUE, REF
from src.sheaf import SheafEdgeScorer, sparse_norm_adj

use_style()

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


def assemble(windows, red, history, split_k, K=COVER_K):
    """Walk windows in time order through ONE continuous history, so novelty is
    always judged against everything earlier (warm-up + train), never reset at the
    split. `history` is mutated and may be pre-seeded for warm-up. Rows from the
    first `split_k` windows are train; the rest are test.

    Returns (Xbase, Xcover, y, is_train, n_test_windows)."""
    Xb, Xc, ys, is_train = [], [], [], []
    for i, w in enumerate(windows):
        Xb.append(baseline_features(w, history))
        Xc.append(cover_features(w, K))
        ys.append(edge_labels(w, red))
        is_train.append(np.full(len(w.edges), i < split_k, dtype=bool))
        for a in w.edges:               # only now does this window become "past"
            history.update(a)
    return (np.concatenate(Xb), np.concatenate(Xc), np.concatenate(ys),
            np.concatenate(is_train), len(windows) - split_k)


def fit_score(Xtr, ytr, Xte):
    # standardise first: the raw features (novelty bits vs degree counts vs reach
    # sizes) live on wildly different scales, which left lbfgs unconverged and the
    # coefficients unstable. A scaler in the pipeline fixes both.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def sheaf_edge_scores(windows, split_k, red, seed=0, epochs=25):
    """Train the end-to-end sheaf edge detector on the train windows and return a score
    for every edge, in the same window order assemble() used (so it aligns with y)."""
    torch.manual_seed(seed)
    model = SheafEdgeScorer(in_dim=2, d=2, h=8, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    cache = []                                   # per window: (P, X, edge_index, labels)
    for w in windows:
        A = w.A
        X = np.stack([np.log1p(A.sum(1)), np.log1p(A.sum(0))], 1).astype(np.float32)
        ei = np.array([[w.index[a.src_comp] for a in w.edges],
                       [w.index[a.dst_comp] for a in w.edges]], np.int64)
        cache.append((sparse_norm_adj(A), torch.from_numpy(X),
                      torch.from_numpy(ei), torch.from_numpy(edge_labels(w, red).astype(np.float32))))

    pos = sum(float(cache[i][3].sum()) for i in range(split_k))
    tot = sum(len(cache[i][3]) for i in range(split_k))
    pw = torch.tensor([(tot - pos) / max(pos, 1.0)], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        model.train()
        for i in range(split_k):
            P, X, ei, lab = cache[i]
            if ei.shape[1] == 0:
                continue
            opt.zero_grad()
            lossf(model(P, X, ei), lab).backward()
            opt.step()

    model.eval()
    out = []
    with torch.no_grad():
        for P, X, ei, lab in cache:
            out.append(np.zeros(0, np.float32) if ei.shape[1] == 0
                       else torch.sigmoid(model(P, X, ei)).numpy())
    return np.concatenate(out)


def _stream_with_warmup(auth_iter, t_start, history):
    """Single pass: events before t_start seed the novelty history (warm-up, not
    scored); events at/after t_start are yielded for windowing. Keeps memory flat
    by never materialising the warm-up events."""
    for a in auth_iter:
        if t_start is not None and a.time < t_start:
            history.update(a)
        else:
            yield a


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
    ap.add_argument("--warmup-days", type=float, default=3.0,
                    help="days before --t-start used to seed novelty history (not scored)")
    args = ap.parse_args()

    history = History()
    if args.smoke:
        print("[smoke test: synthetic slice -- plumbing only, not evidence]\n")
        auths, red, window_seconds = synthetic_slice(window_seconds=args.window)
        windows = list(build_windows(iter(auths), window_seconds))
    else:
        if not (args.auth and args.redteam):
            ap.error("provide --auth and --redteam, or use --smoke")
        red = load_redteam(args.redteam)
        warm_start = None
        if args.t_start is not None:
            warm_start = max(0, args.t_start - int(args.warmup_days * 86400))
        events = iter_auth(args.auth, warm_start, args.t_end)
        # one pass: warm-up events seed `history`, window events get binned
        windows = list(build_windows(
            _stream_with_warmup(events, args.t_start, history), args.window))
        print(f"warm-up seeded history: {len(history.edges):,} edges, "
              f"{len(history.user_dst):,} (user,host) pairs\n")

    split_k = int(len(windows) * args.train_frac)
    Xb, Xc, y, is_train, n_te_win = assemble(windows, red, history, split_k)
    Xb_tr, Xb_te = Xb[is_train], Xb[~is_train]
    Xc_tr, Xc_te = Xc[is_train], Xc[~is_train]
    ytr, yte = y[is_train], y[~is_train]
    n_days = max(1, (n_te_win * args.window) / 86400)

    if yte.sum() == 0 or ytr.sum() == 0:
        print("No red-team positives in a split -- widen the slice "
              "(--t-start/--t-end) or shift --train-frac to put positives on both sides.")
        return

    # src_reach is cover column 0 -- the "how far can this source already move"
    # scalar. A third model isolates it, to test whether the whole cover gain is
    # just that one reachability feature (i.e. pivot fan-out) rather than the block.
    sr = COVER_NAMES.index("src_reach_K")
    runs = {
        "baseline (novelty+degree)": (Xb_tr, Xb_te),
        "baseline + src_reach only": (np.hstack([Xb_tr, Xc_tr[:, sr:sr + 1]]),
                                      np.hstack([Xb_te, Xc_te[:, sr:sr + 1]])),
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

    # an actual learned sheaf NN, end-to-end on the windowed access graphs (no novelty
    # features, purely structural), to see whether a topological model beats the bar
    sh = sheaf_edge_scores(windows, split_k, red)[~is_train]
    results["sheaf NN (end-to-end)"] = {
        "pr_auc": average_precision_score(yte, sh),
        "recall": recall_at_fpr(yte, sh),
        "alerts": alerts_per_day(yte, sh, n_days),
        "score": sh,
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

    # mechanism diagnostic: which features drive the full baseline+cover model?
    # If a reachability feature (esp. src_reach) dominates, the gain is pivot
    # fan-out detection, not novelty -- the structural signal this repo is about.
    scaler = StandardScaler().fit(np.hstack([Xb_tr, Xc_tr]))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(scaler.transform(np.hstack([Xb_tr, Xc_tr])), ytr)
    names = BASELINE_NAMES + COVER_NAMES
    print("\ntop feature drivers (standardized |coef|, + = more suspicious):")
    for nm, co in sorted(zip(names, clf.coef_[0]), key=lambda kv: -abs(kv[1]))[:6]:
        tag = "  <- cover" if nm in COVER_NAMES else ""
        print(f"  {nm:20s} {co:+.2f}{tag}")

    print("\nNote: red-team labels are a lower bound; some 'false positives' may be "
          "unlabelled malicious, so precision/alerts are pessimistic.\n")

    _plot_pr(yte, results, base_rate)
    print("Wrote results/lanl_pr.png")


def _plot_pr(y, results, base_rate):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    palette = {"baseline (novelty+degree)": REF, "baseline + cover": BLUE}
    for name, r in results.items():
        p, rc, _ = precision_recall_curve(y, r["score"])
        ax.plot(rc, p, lw=2, color=palette.get(name, "#9bb8d3"),
                label=f"{name} (AP={r['pr_auc']:.3f})")
    ax.axhline(base_rate, ls="--", lw=1, color="#b0b6bd",
               label=f"base rate ({base_rate:.3%})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Does the cover block beat novelty + degree?")
    ax.legend(loc="upper right")
    fig.savefig("results/lanl_pr.png"); plt.close(fig)


if __name__ == "__main__":
    main()
