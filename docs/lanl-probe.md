# Real-data probe (LANL)

```bash
python run_lanl.py --smoke                                  # synthetic plumbing test
python run_lanl.py --auth auth.txt.gz --redteam redteam.txt # the real slice
```

The static and temporal demos are synthetic, so I choose the task and can always build
one where the cover wins. That teaches a mechanism but proves nothing about practice.
This probe tests the idea on real data, and the hypothesis is written down before any
number is trusted, so a negative result is reported rather than dropped.

The data is LANL's Comprehensive, Multi-Source Cyber-Security Events (Kent, 2015): 58
days of authentication events with a small set of labelled red-team logons. Download it
yourself from [csr.lanl.gov](https://csr.lanl.gov/data/cyber1/); the harness does not
fetch it. The task is edge-level: given an authentication `src -> dst` in a time window,
was it red-team lateral movement? The split is temporal, training on early days and
testing on later ones, and novelty is only ever computed against the past.

The baseline is the honest part. A real authentication graph is essentially one giant
connected component, so counting components or measuring global reachability, the trick
that scores 1.0 on the synthetic task, is uninformative here. What works on LANL is
whether a `src -> dst` edge or a credential-on-a-host has been seen before, since
red-team movement so often lands on a host the credential has never touched. That
novelty-plus-degree detector, with no GNN, is the bar. The probe fits the same logistic
regression twice, on baseline features and on baseline features plus a bounded,
ego-net-restricted reachability cover, and asks whether the cover adds recall at a
realistic false-positive budget (0.1% FPR), alongside an alerts-per-day figure.

For novelty to mean anything, two things matter. The history that decides "seen before"
runs continuously across the train/test split, so test edges are judged against
everything earlier, never reset. And the days just before the window seed that history
without being scored (a warm-up), so the first window is not flooded with spurious
"novel" edges. Both are handled by the harness (`--warmup-days`, default 3).

## Hypothesis, pre-registered

I expected the novelty baseline to be hard to beat, and that the cover would add little
once the signal is not handed over by construction. If the cover cleared the baseline at
low FPR, that would be a real result; if not, the null is the finding. One caveat
colours every number: the red-team labels are a lower bound, so some apparent false
positives may be unlabelled malicious activity, making precision pessimistic.

## Result: the pre-registered null holds

The campaign in this data is essentially single-pivot: host `C17693` is the source for
273/273 red-team events on day 8 and 277/290 on days 12-13. So this tests temporal
robustness, not cross-pivot generality.

Day 8 (train the morning, 28 positives; test the afternoon, 246 positives):

```
model                       PR-AUC   recall@0.1%FPR   alerts/day
baseline (novelty+degree)    0.04         0.53          3710
baseline + cover             0.16         0.93          3810
```

On day 8 the cover appeared to help. But one window with 28 training positives is thin,
and the operating point is unusable: 0.1% FPR on 3.6M edges is ~3,800 alerts/day, which
no analyst would run. So it needed replication.

Day 12 (train the morning, 129 positives; test the afternoon, 80 positives):

```
model                       PR-AUC   recall@0.1%FPR   alerts/day
baseline (novelty+degree)    0.011        0.55          3947
baseline + src_reach only    0.005        0.25          3912
baseline + cover             0.000        0.01          3859
```

It did not replicate. With more training positives and a cleaner split, the cover adds
nothing and in fact hurts (recall 0.55 to 0.01). So across two windows the verdict is the
pre-registered null: on real data, where the signal is not handed over by construction,
the reachability cover does not beat a no-GNN novelty baseline. The day-8 result was a
one-window artifact, which is exactly why replication mattered.

Caveats, all pushing the same way:

- The operating point (0.1% FPR, about 3,900 alerts/day) is not usable; a real budget is
  tens per day. We never re-scored at one.
- Single-pivot campaign, so cross-pivot generality is untested.
- These runs used logistic regression whose lbfgs did not fully converge on the unscaled
  features; the harness now standardises features. That does not change the qualitative
  null, since the day-12 cover collapse is far too large for a conditioning artifact.

What survives regardless of the verdict: the harness improvements are real keepers.
Novelty with warm-up and continuous history, and a sparse `reachable_counts` (scipy) that
took the per-window cover from dense O(n^3) to O(nnz*K), which is what made a real run
over ~4,000-node-per-window graphs tractable at all.

## Running the real slice

1. Download `auth.txt.gz` and `redteam.txt.gz` from
   [csr.lanl.gov/data/cyber1](https://csr.lanl.gov/data/cyber1/) (email-gated form).
2. Pick a red-team-active span from `redteam.txt`. The activity clusters on day 8 (273
   labelled events), so days 7-9 straddle the spike and put positives on both sides of
   a temporal split.
3. Run (the file is time-sorted, so the loader streams only up to `--t-end`):

   ```bash
   # day 8 (the spike); --t-start 691200 --t-end 777600 is a 1-day variant
   python run_lanl.py --auth auth.txt.gz --redteam redteam.txt.gz \
     --t-start 604800 --t-end 864000 --window 3600 --warmup-days 3

   # day 12, the replication window above
   python run_lanl.py --auth auth.txt.gz --redteam redteam.txt.gz \
     --t-start 1036800 --t-end 1123200 --window 3600 --warmup-days 3
   ```

   The runner reports if a split has no positives and asks you to widen the slice or
   shift `--train-frac`. Reaching a window means streaming the gzip up to it (a few
   minutes); the cover math itself is seconds.
