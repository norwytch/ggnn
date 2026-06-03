# Cover-Based GNNs for Lateral-Movement Detection — a proof of concept

This repo is my attempt to understand Grothendieck Graph Neural Networks (GGNNs)
and apply them to a security use case. It is a small proof of concept on synthetic
data, not a deployable detector.

BLUF: Out-of-the-box GNNs can learn over networks, but cannot recover the blast radius
signal a lateral-movement attacker would exploit. Grothendieck-style covers can.

| model | PR-AUC | recall @ 1% FPR |
|---|---|---|
| GCN, GIN (1-WL) | ~0.03 (= base rate) | 0 |
| LogReg on # components | 1.0 | 1.0 |
| GCN + reachability feature | 1.0 | 1.0 |
| CoverNet (cover features) | 1.0 | 1.0 |

*~3% positives, tested on an unseen graph size. The **signal**, not the
architecture, is what matters: once reachability is exposed, even logistic
regression on a single scalar nails it.*

## Use case

Networks can be represented by graphs, called access graphs. Nodes/vertices are hosts, end devices, or user accounts, and a directed edge `u → v` represents that an identity on `u` can authenticate to `v`.
We classify two estates that look identical locally but differ in blast radius:

| class | structure | training | meaning |
|------|-----------|----------|---------|
| `flat` | one directed *n*-cycle (e.g. a 12-cycle) | sizes *n* vary; tested on a size never seen in training | every host reaches every other — compromise one, reach all |
| `segmented` | *k* disjoint directed cycles (e.g. two 6-cycles) | sizes *n* and segment count *k* vary; tested on a size never seen in training | isolated enclaves — compromise one, reach only its enclave |

Telling these two estates apart is critical in the context of lateral movement. The flat estate will endanger the entire network if compromised, whereas the segmented estate will endanger only its local segment (enclave). So let's get into why a standard GNN cannot tell these two apart.

## Why standard GNNs fail: the 1-WL test

Message-passing GNNs are bounded by the **1-Weisfeiler-Leman (1-WL)** test. In both
classes every node has in-degree 1, out-degree 1, and the same feature, so 1-WL
gives every node the same colour forever — and a flat estate and a segmented one
end up with the same embedding. That's why GCN and GIN score at the base rate.

The missing signal is global responsibility: the fraction of the estate each node
can reach (~1.0 when flat, ~1/k when split into k enclaves), or equivalently the
number of connected components (1 vs k). It's a property of the whole graph, not of
any neighbourhood, so a 1-WL model can't recover it.

## So why Grothendieck?

GGNNs generalise a node's neighbourhood to a cover: a family of directed paths
written as matrices, where matrix multiplication is path concatenation. `CoverNet`
uses one small instance — the reachability cover up to `K` hops — exposing, per node:

* arriving walks of length `t` (`Aᵗ · 1`),
* closed walks of length `t` (`diag(Aᵗ)`) — the cycle signal 1-WL misses,
* the size of its reachable set within `K` hops — the **blast radius**.

An MLP over these solves the task exactly. The paper goes further and claims the
same idea scales, with cover/sieve operators separating graphs that defeat even
3-WL. Treat that as the paper's claim rather than ours: it is the strongest
expressivity assertion in the now-withdrawn submission, and a reviewer disputed it
directly, arguing the framework is bounded by 2-FWL — which, since 2-FWL is
equivalent to 3-WL, would contradict the "beyond 3-WL" reading. What this repo
actually demonstrates is narrower and self-contained: the separations below, which
hold regardless of where the framework lands in the WL hierarchy.

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py
```

Runs in under a minute on CPU. Prints a results table and writes
`results/pr_auc.png` and `results/separation.png`.

## Guided notebooks

If you'd rather read than run, [`notebooks/`](notebooks/) is a four-part tour —
covers → static → sieve → temporal — building each idea from scratch with inline
math and plots. Start with [01_covers](notebooks/01_covers.ipynb); they're committed
with outputs rendered, so they read without running anything. Kernel setup is in
[notebooks/README.md](notebooks/README.md).

## The sieve cover

```bash
python run_sieve.py
```

The reachability cover is genuine GGNN machinery, but on the task above it does no work that a simpler method couldn't do, since a plain count of connected components ties it. So for this demo let's pick a task where ordinary covers provably fail.

The task is to distinguish the 4×4 Rook's graph from the Shrikhande graph. Both of these are strongly regular with the same parameters, SRG(16,6,2,2), they are non-isomorphic, and they are indistinguishable by the WL test. They are also cospectral, which means their walk and closed-walk counts agree at every length. The script prints `Tr(Aᵗ) = 0, 96, 192, 1536, 7680` for both graphs, and because those numbers are identical, any model built on walks or reachability is stuck at chance.

The sieve cover takes a different view. To see how, it helps to recall what the earlier covers actually do. The walk and reachability covers are built from the directed paths that leave a node, which is to say powers of the adjacency matrix `A`, and the features they expose are all outward-facing: how many walks arrive, how many close up, and how much of the estate the node can reach. That is a global question, namely *where can I get to*. The catch is that on a regular graph every node has the same degree, so these walk counts come out constant across nodes and identical between two cospectral graphs, which is exactly why they cannot tell Rook from Shrikhande.

The sieve cover asks a local question instead. Rather than following paths out of a node, it looks sideways at the structure among the node's neighbours, that is, the subgraph those neighbours induce on each other. Concretely it restricts `A` to just the neighbour set and measures the wiring inside it, counting triangles, connected components, and short closed walks within that little subgraph. So the contrast is *not where my neighbours lead, but how my neighbours are wired to one another*. In the Rook's graph that induced subgraph is two triangles, whereas in the Shrikhande graph it is a single 6-cycle. This is the same C₆-versus-2·C₃ motif we saw before, except that now it shows up as a local property of each node rather than a global one. In the GGNN framing both are still covers fed through the same `Tr` homomorphism; the sieve cover is simply a more refined choice that conditions on the structure among the elements covering a node instead of treating them as a flat set.

![sieve](results/sieve_acc.png)

`WalkCoverNet` and the trivial baseline both sit at chance, whereas `SieveNet`, which is the same architecture with the sieve cover added, reaches 100%. The entire gap comes from that one choice of cover. This is the GGNN thesis in practice, that you design message passing by choosing covers, made load-bearing on a pair of graphs that defeats WL.

It is worth being honest about what this separation is and is not. The sieve cover
works here by counting hand-chosen substructures in each node's neighbourhood, namely
triangles, components, and short closed walks, and separating cospectral SRGs by
neighbourhood substructure count is essentially the known Graph Substructure Network
(GSN) result, which we cite in Related Work. So the win is real but not new
expressivity in disguise: a reviewer of the paper made exactly this point, that the
operations amount to substructure counting rather than an enlarged operation space.
What the demo does show cleanly is the GGNN framing's central move — that swapping one
cover for another, with the architecture held fixed, is what flips the model from
chance to 100%.

*(The sieve cover here is one principled instantiation of a precomposition-refined cover. It is not a verbatim copy of the paper's construction, which public sources under-specify.)*

## When time is the signal

```bash
python run_temporal.py
```

The static demos show a *model* limit. This one shows a *representation* limit:
collapse a stream of timestamped events into a static graph and you throw away the
ordering — and ordering is the signal.

Both classes share an **identical static graph**. Every stream has the same chain
`f → … → g`, the same background traffic, and an exfiltration event. Only the chain's
*timestamps* differ:

* **attack** — chain edges fire in increasing order, so a time-respecting path
  `f → g` exists, and exfil follows.
* **benign** — the same edges fire scrambled, so no time-respecting path exists.

Since the static graph is identical, any static model is stuck at the base rate. The
fix is a **temporal cover**: time-respecting reachability from a single time-ordered
sweep (the temporal analogue of summing powers of `A`). Categorically, that sweep is
the **causal past** of an event.

![temporal PR-AUC](results/temporal_pr_auc.png)

A static GCN and an MLP on static-reachability features both sit at the ~7% base
rate; the same MLP on **temporal** cover features reaches PR-AUC ~0.83. Not perfect,
and that's honest: the leftover false positives are benign streams whose scrambled
chain happens to land in order (~1/5! of the time) — something only event
correlation could clean up.

![lead time](results/temporal_lead.png)

Time also gives a metric a static model can't even define: how long before exfil the
path completes. Every detected attack gets positive lead time — caught before the
data leaves.

## Real-data probe (LANL)

```bash
python run_lanl.py --smoke                                  # synthetic plumbing test
python run_lanl.py --auth auth.txt.gz --redteam redteam.txt # the real slice
```

Everything above is synthetic, and on synthetic data I get to choose the task, which
means I can always build one where the cover wins. That is fine for teaching a
mechanism but it proves nothing about whether covers are worth their cost in
practice. This probe is the attempt to break that circularity on real data, and I am
writing down the hypothesis here *before* I trust any number, so that a negative
result is reported as a negative result rather than quietly dropped.

The data is LANL's *Comprehensive, Multi-Source Cyber-Security Events* (Kent, 2015),
which is 58 days of real authentication events with a small set of labelled red-team
logons. You have to download it yourself from
[csr.lanl.gov](https://csr.lanl.gov/data/cyber1/); the harness does not fetch it. The
task is edge-level: given an authentication `src → dst` in a time window, was it
red-team lateral movement? The split is temporal, so the model trains on early days
and is tested on later ones, and novelty is only ever computed against the past.

The honest part is the baseline. A real authentication graph is essentially one giant
connected component, so the trick that scores a perfect 1.0 on the synthetic task —
counting components or measuring global reachability — is simply uninformative here.
What actually works on LANL is mundane: whether a `src → dst` edge or a
credential-on-a-host has ever been seen before, because red-team movement so often
lands on a host the credential has never touched. That novelty-plus-degree detector,
with no GNN at all, is the bar to beat. The probe fits the *same* logistic regression
twice, once on those baseline features and once on baseline features plus a bounded,
ego-net-restricted reachability cover, and asks whether the cover block adds recall at
a realistic false-positive budget (0.1% FPR), reported alongside an alerts-per-day
figure that an analyst would actually feel.

**Hypothesis, pre-registered.** I expect the novelty baseline to be hard to beat, and
I will be unsurprised if the cover adds little or nothing once the signal is no longer
handed over by construction. If the cover does clear the baseline at low FPR, that is
a genuine result worth building on; if it does not, that null is the finding, and it
is exactly what the circularity caveat in the limitations predicts. One caveat colours
every number: the red-team labels are a lower bound, so some apparent false positives
may be unlabelled malicious activity, which makes the reported precision pessimistic.
As of now the real slice has not been run; only the synthetic smoke test has, and it
behaves as intended by reporting a null when the attacks are already caught by
novelty.

## Repository layout

```
src/data.py        access-graph generator: imbalanced, variable size,
                   train/test split with an unseen size (generalization probe)
src/operators.py   1-hop propagators, reachability-cover features, components count
src/models.py      GCN, GIN (1-WL); GCNPlus (GCN + cover feats); CoverNet
run_experiment.py  static demo: + trivial LogReg-on-#components baseline; PR-AUC
                   and recall@1%FPR over 3 seeds; renders both figures
src/covers.py      explicit cover algebra: Tr homomorphism, walk / reachability
                   / sieve covers
src/srg_data.py    Rook's vs Shrikhande SRG pair (WL-indistinguishable, cospectral)
src/sieve_models.py  cover-combining models with learnable per-cover gates
run_sieve.py       sieve demo: only adding the sieve cover separates the SRG pair
src/temporal_*.py  temporal demo: event-stream generator, time-respecting
                   reachability operator, and models
run_temporal.py    temporal demo runner (static models vs temporal cover) +
                   an early-detection (lead-time) metric
src/lanl.py        LANL loader: auth/redteam parsing, time-windowed access
                   graphs, + a synthetic slice for a no-download smoke test
src/baselines.py   honest baseline (edge/credential novelty + degrees) vs the
                   bounded ego-net reachability cover, as per-edge features
run_lanl.py        real-data probe: temporal split, same classifier on baseline
                   vs baseline+cover; PR-AUC, recall@0.1%FPR, alerts/day
notebooks/         four-part guided tour (covers -> static -> sieve -> temporal),
                   each importing from src/ with rendered outputs
```

## Limitations and honest scope

A minimal demonstration of one mechanism, **not** a deployable detector:

* **Structure must be the signal.** The edge exists precisely because the classes
  are 1-WL-equivalent. When node/edge features already carry the signal (ports,
  byte counts, timing), a plain GCN can match this and the advantage disappears.
* **Bounded for tractability.** Reachability operators trend toward dense, ~O(n³)
  computation. Here `K` is small and graphs are tiny; real estates would need
  sparse / ego-net-restricted covers — exactly where the gain could erode.
* **Static and synthetic.** Real telemetry is streaming, time-ordered, typed, and
  heavily imbalanced. None of that is modelled here.
* **Robustness untested.** An attacker can pad benign-looking hops; a more
  expressive operator isn't automatically a more robust one.
* **Leans on a withdrawn, disputed source.** The framework comes from a withdrawn
  ICLR 2026 submission whose strongest expressivity claim a reviewer rejected, and the
  sieve separation is essentially the known GSN substructure-counting result. The
  demos here stand on their own, but the broader "beyond 3-WL" framing does not — see
  the sieve section and References for the full hedge.

## Possible extensions

* Time-ordered covers for provenance-graph APT detection — the natural next domain.
* A real-data sanity check on a small slice of the LANL authentication dataset
  around labelled red-team events.
* Typed/heterogeneous covers per edge type; sparse operators for scale.

## Related work — where this sits

This PoC sits at the intersection of three literatures that rarely cite each other.
It isn't a new state of the art — it's a bridge between them.

**Expressivity beyond 1-WL.** Standard MPNNs are bounded by the 1-WL test (Xu et al.,
*GIN*, 2019; Morris et al., *k-GNNs*, arXiv:1810.02244), and the logic/WL
correspondence is now well understood (Grohe, *The Logic of GNNs*, arXiv:2104.14624).
The families that break the ceiling — higher-order k-WL networks, subgraph GNNs
(Bouritsas et al., *GSN*, arXiv:2006.09252; Bevilacqua et al., *ESAN*, 2022), and
substructure/positional encodings — all share the tension this repo faces too:
**expressivity trades off against scalability**. Recent work chases cheap expressivity
(e.g. invariant-stratified propagation, Hevapathige et al., arXiv:2603.01388, KDD
2026). The SRG/CSL/BREC isomorphism benchmarks are the standard yardsticks, and GSN
already reported strong SRG results given domain-chosen substructures.

**Topological & categorical deep learning** is where the *cover* idea belongs: sheaf
neural networks (Hansen–Ghrist; Bodnar et al., *Neural Sheaf Diffusion*, 2022),
simplicial/cellular networks, and topological deep learning over combinatorial
complexes (Hajij et al., 2023). The live trend is unification — *Copresheaf
Topological Neural Networks* (arXiv:2505.21251) subsumes GNNs, attention, sheaf nets,
and TNNs under one formalism, the broader current the Grothendieck framework swims in.

**GNN-based intrusion / APT detection** is active but driven by different concerns —
temporal modelling, false-positive rates, reproducibility. Representative systems:
MAGIC (USENIX Security 2024), Kairos (arXiv:2308.05034, temporal GNN over
provenance), Slot (CCS 2025, graph RL), CONTINUUM (arXiv:2501.02981, spatio-temporal),
and JBEIL, which targets **lateral movement** specifically. A 2025 reproducibility
study (ACM REP '25) documents how brittle several of these pipelines are. Notably,
none frames detection as a 1-WL-expressivity problem — the gap this PoC points at.

**Positioning.** The detection community thinks temporally but not in
cover/expressivity terms; the expressivity community rarely touches authentication or
provenance graphs. This repo makes the bridge explicit on a minimal example: a task
1-WL provably can't solve, that a reachability cover resolves by construction. The
natural next step — a *time-ordered* cover aligned with the TGN-style systems above —
is where the genuine whitespace lies.

## References

* Paper: *Grothendieck Graph Neural Networks Framework: An Algebraic Platform for
  Crafting Topology-Aware GNNs* — arXiv:2412.08835. An ICLR 2026 submission of this
  work was later withdrawn, so treat its claims as an unrefereed preprint. We know
  this because the paper's OpenReview page
  ([forum 4hXoo5MhxZ](https://openreview.net/forum?id=4hXoo5MhxZ)) carries the status
  label "ICLR 2026 Conference Withdrawn Submission" (last modified 2025-11-20). Note
  that the arXiv preprint itself has not been retracted and remains available; it is
  the conference submission that was withdrawn.
* The `flat` vs `segmented` construction is the security reading of the classic
  C₆ vs 2·C₃ counterexample to 1-WL.

*Built as a portfolio proof of concept. Synthetic data only; no real network
telemetry is used.*
