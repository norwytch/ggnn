"""
Sieve cover demo: making the GGNN aspect load-bearing.

    python run_sieve.py

Task: distinguish the 4x4 Rook's graph from the Shrikhande graph -- a pair of
non-isomorphic SRG(16,6,2,2) graphs that defeat the Weisfeiler-Leman test and are
cospectral. The walk/reachability covers (and any model built on them) are at
chance by construction; adding the *sieve cover* to the basis separates them.
Writes results/sieve_acc.png.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.srg_data import make_dataset, rook44, shrikhande
from src.covers import walk_cover, reachability_cover, sieve_cover, check_homomorphism
from src.sieve_models import CoverCombiner

K, L, HIDDEN, EPOCHS, BATCH, SEEDS = 6, 6, 32, 80, 64, (0, 1, 2)


def blocks_for(A):
    return [torch.from_numpy(walk_cover(A, K)),
            torch.from_numpy(reachability_cover(A, K)),
            torch.from_numpy(sieve_cover(A, L))]


def collate(items, idx, use):
    n = items[idx[0]][0].shape[0]
    nb = len(use)
    out = [[] for _ in range(nb)]; bt = []
    for j, i in enumerate(idx):
        for bi, b in enumerate(use):
            out[bi].append(items[i][b])
        bt.append(torch.full((n,), j, dtype=torch.long))
    return [torch.cat(o) for o in out], torch.cat(bt), len(idx)


def train_eval(use, dims, items, y, tr, te, seed):
    torch.manual_seed(seed)
    model = CoverCombiner(dims, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(); yt = torch.from_numpy(y)
    for _ in range(EPOCHS):
        model.train(); perm = np.random.permutation(tr)
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            blk, bt, num = collate(items, idx, use)
            opt.zero_grad(); lossf(model(blk, bt, num), yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        blk, bt, num = collate(items, te, use)
        acc = (model(blk, bt, num).argmax(1) == yt[te]).float().mean().item()
    return acc, model.gate_weights()


def main():
    print("\nTr is a monoid homomorphism (cover composition -> matrix product):",
          check_homomorphism())
    # cospectral / walk-cover-blind check
    wr, ws = walk_cover(rook44(), K), walk_cover(shrikhande(), K)
    cwt = lambda A: [int(round(np.trace(np.linalg.matrix_power(A, t)))) for t in range(1, 6)]
    print("Closed walks Tr(A^t), t=1..5  -> Rook:", cwt(rook44()),
          " Shrikhande:", cwt(shrikhande()), " (cospectral: identical)")
    print("Walk-cover node-feature multisets identical (Rook vs Shrikhande):",
          np.array_equal(np.sort(wr, 0), np.sort(ws, 0)))
    print("Sieve-cover #triangles per node  -> Rook:", set(sieve_cover(rook44(), L)[:, 0].astype(int)),
          " Shrikhande:", set(sieve_cover(shrikhande(), L)[:, 0].astype(int)), "\n")

    As, y = make_dataset(n_per_class=300, seed=0)
    rng = np.random.default_rng(0); order = rng.permutation(len(y))
    cut = int(0.7 * len(y)); tr, te = order[:cut], order[cut:]

    raw = [[walk_cover(A, K), reachability_cover(A, K), sieve_cover(A, L)] for A in As]
    stats = []                                   # standardize each block on TRAIN nodes
    for bi in range(3):
        train_rows = np.concatenate([raw[i][bi] for i in tr], 0)
        mu, sd = train_rows.mean(0), train_rows.std(0) + 1e-6
        stats.append((mu, sd))
    items = [[torch.from_numpy(((rb[bi] - stats[bi][0]) / stats[bi][1]).astype(np.float32))
              for bi in range(3)] for rb in raw]
    dims = [b.shape[1] for b in items[0]]

    print(f"{'model (cover basis)':30s} {'test acc':>9s}")
    print("-" * 42)
    results = {}

    # trivial baseline: LogReg on mean walk-cover features (provably blind)
    feat = np.array([walk_cover(A, K).mean(0) for A in As])
    lr = LogisticRegression(max_iter=500).fit(feat[tr], y[tr])
    results["LogReg / walk cover"] = ((lr.predict(feat[te]) == y[te]).mean(), None)

    cfgs = {
        "WalkCoverNet {walk,reach}":   ([0, 1], [dims[0], dims[1]]),
        "SieveNet {walk,reach,SIEVE}": ([0, 1, 2], dims),
    }
    for name, (use, d) in cfgs.items():
        accs, gates = zip(*[train_eval(use, d, items, y, tr, te, s) for s in SEEDS])
        results[name] = (float(np.mean(accs)), np.mean(gates, 0))

    for name, (acc, extra) in results.items():
        print(f"{name:30s} {acc:>9.3f}")
    sg = results["SieveNet {walk,reach,SIEVE}"][1]
    print(f"\nSieveNet learned cover gates [walk, reach, sieve] = "
          f"[{sg[0]:.2f}, {sg[1]:.2f}, {sg[2]:.2f}]  (it leans on the sieve cover)\n")

    _plot({k: v[0] for k, v in results.items()})
    print("Wrote results/sieve_acc.png")


def _plot(accs):
    names = list(accs); vals = [accs[n] for n in names]
    colors = ["#9aa0a6", "#9aa0a6", "#1a73e8"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0.5, ls="--", lw=1, color="#d93025", label="chance (0.5)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Rook vs Shrikhande (cospectral SRGs, WL-indistinguishable):\nonly adding the sieve cover to the basis separates them")
    ax.legend(loc="center right"); fig.tight_layout()
    fig.savefig("results/sieve_acc.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
