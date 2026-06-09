# When time is the signal

```bash
python run_temporal.py
```

The static demos show a model limit. This one shows a representation limit: collapse a
stream of timestamped events into a static graph and you throw away the ordering, which
is the signal.

Both classes share an identical static graph. Every stream has the same chain
`f -> ... -> g`, the same background traffic, and an exfiltration event. Only the chain's
timestamps differ:

- attack: chain edges fire in increasing order, so a time-respecting path `f -> g`
  exists and exfil follows.
- benign: the same edges fire scrambled, so no time-respecting path exists.

Since the static graph is identical, any static model is stuck at the base rate. The fix
is a temporal cover: time-respecting reachability from one time-ordered sweep, the
temporal analogue of summing powers of `A`. Categorically that sweep is the causal past
of an event.

![temporal PR-AUC](../results/temporal_pr_auc.png)

A static GCN and an MLP on static-reachability features both sit at the ~7% base rate.
The same MLP on temporal cover features reaches PR-AUC ~0.83. The leftover false
positives are benign streams whose scrambled chain happens to land in order (about 1/5!
of the time), which only event correlation could clean up.

![lead time](../results/temporal_lead.png)

Time also gives a metric a static model cannot define: how long before exfil the path
completes. Every detected attack gets positive lead time.

Like the static demos, this task is built so the temporal signal is the answer. It
illustrates a representation limit; it is not evidence that a temporal cover beats an
existing temporal-GNN system (see [related work](related-work.md)).
