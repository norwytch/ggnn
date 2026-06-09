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

## Hypothesis, pre-registered

I expect the novelty baseline to be hard to beat and will be unsurprised if the cover
adds little once the signal is not handed over by construction. If it clears the
baseline at low FPR, that is a real result; if not, the null is the finding. One caveat
colours every number: the red-team labels are a lower bound, so some apparent false
positives may be unlabelled malicious activity, making precision pessimistic.

The real slice has not been run. Only the synthetic smoke test has, and it reports a null
when the attacks are already caught by novelty:

```
baseline (novelty+degree)       PR-AUC 0.908   recall@0.1%FPR 0.91   33 alerts/day
baseline + cover                PR-AUC 0.902   recall@0.1%FPR 0.91   33 alerts/day
verdict: covers do NOT clear the baseline here -- honest null
```

That table is synthetic and is plumbing, not evidence.

## Running the real slice

1. Download `auth.txt.gz` and `redteam.txt` from
   [csr.lanl.gov/data/cyber1](https://csr.lanl.gov/data/cyber1/).
2. Find a red-team-active span from `redteam.txt`.
3. Run so both splits contain positives:

   ```bash
   python run_lanl.py --auth auth.txt.gz --redteam redteam.txt \
     --t-start <start> --t-end <end> --window 3600
   ```

   The runner reports if a split has no positives and asks you to widen the slice.
