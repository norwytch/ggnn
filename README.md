# Cover-Based GNNs for Lateral-Movement Detection — a proof of concept

A small, fully reproducible demonstration of a **structural blind spot** of
1-Weisfeiler-Leman (1-WL) graph neural networks on a lateral-movement task — and
of the fact that the missing signal is *global reachability*, recoverable several
ways, one of which is the **Grothendieck GNN (GGNN)** notion of a *cover*.

This is a teaching proof of concept, not a new detector and not a benchmark win.
The honest claim it supports: a defender cares about reachability / blast radius;
plain message-passing GNNs are blind to it by construction; and once the signal
is exposed, even logistic regression on a single scalar recovers it. The point is
the blind spot and why it matters — not that any one model is superior.

![PR-AUC](results/pr_auc.png)

> Evaluated under realistic class imbalance (~3% positives) on an **unseen graph
> size**, with security-appropriate metrics (PR-AUC, recall at a fixed 1% false-
> positive rate). GCN and GIN sit exactly at the base rate (PR-AUC ~ 0.03,
> recall@1%FPR = 0): they cannot see global reachability. A trivial baseline
> (logistic regression on the number of connected components), a plain GCN *given*
> the reachability features, and the cover-based `CoverNet` all reach PR-AUC = 1.0.
> The lesson is that the **signal**, not the architecture, is what matters here —
> stated plainly because pretending otherwise is what a reviewer would catch first.

## The problem, in security terms

Each graph is a tiny **access graph**: a node is a host/account, a directed edge
`u → v` means "an identity on `u` can authenticate to `v`". We classify two
estates that are *locally identical* but differ in the property a defender
actually cares about — **blast radius**:

| class | structure | meaning |
|------|-----------|---------|
| `flat` | one directed 12-cycle | every host eventually reaches every other; compromise one → reach all |
| `segmented` | k disjoint directed cycles | the estate is split into isolated enclaves; compromise one → reach only its enclave |

(The example uses one 12-cycle vs two 6-cycles; the dataset varies graph size
and segment count, and tests on an unseen size.)

Telling these apart is the essence of lateral-movement risk assessment. Yet a
standard GNN cannot do it.

## Why standard GNNs fail here (the interesting part)

Message-passing GNNs are bounded in power by the **1-Weisfeiler-Leman (1-WL)**
test. In *both* classes every node has in-degree 1, out-degree 1, and the same
input feature, so 1-WL assigns every node the same colour forever. A
1-WL-bounded network therefore produces the **same** graph embedding for a flat
estate and a segmented one. In the imbalanced evaluation this shows up as GCN
and GIN scoring exactly at the base rate.

The distinguishing signal is **global reachability**, which 1-hop aggregation
never sees. A size-robust way to read it is the *normalised blast radius* — the
fraction of the estate each node can reach: ~1.0 in a flat estate, ~1/k in a
k-segment one. (Equivalently, the number of connected components: 1 vs k.) These
are functions of the whole graph, not of any node's neighbourhood, so a
1-WL-bounded model cannot recover them — the failure is information-theoretic.

![separation](results/separation.png)

## The fix: a bounded reachability *cover*

The GGNN framework generalises a node's "neighbourhood" to a **cover** — a family
of directed paths translated into matrix form, where matrix multiplication is
path concatenation. `CoverNet` realises a small, sparse instance: the bounded
reachability cover up to `K` hops. For each hop length `t` it exposes, per node,

* the number of length-`t` walks arriving (`Aᵗ · 1`),
* the **closed walks** of length `t` through the node (`diag(Aᵗ)`) — the cycle
  signal 1-WL is blind to, and
* the size of the node's reachable set within `K` hops — its **blast radius**.

An MLP over these features plus the shared readout head solves the task exactly.
This mirrors, at toy scale, the paper's result that cover/sieve operators
distinguish graphs (e.g. strongly regular graphs) that defeat 1-WL and even 3-WL.

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py
```

Runs in under a minute on CPU. Prints a results table and writes
`results/pr_auc.png` and `results/separation.png`.

## Guided notebooks

If you'd rather read than run, [`notebooks/`](notebooks/) is a four-part tour that
builds the ideas from scratch — what a cover *is*, the `Tr` homomorphism, and the
walk / reachability / sieve covers — then walks through each demo with inline math,
plots, and the category-theory intuition behind the sieve. Start with
[01_covers](notebooks/01_covers.ipynb). They're committed with outputs rendered, so
they read without running anything; see [notebooks/README.md](notebooks/README.md)
for the kernel setup.

## The sieve cover: making the GGNN aspect load-bearing

`python run_sieve.py`

The reachability cover above is real GGNN machinery — covers translated to
matrices, composition equal to path concatenation (verified by
`covers.check_homomorphism`, which checks Tr(D1 ∘ D2) = Tr(D1) Tr(D2)). But the
honest baselines show it does no *unique* work on the reachability task: a
component count ties it. To make the framework earn its keep, this demo uses a
task where the plain covers **provably** fail.

Task: distinguish the 4×4 **Rook's graph** from the **Shrikhande graph** — both
SRG(16,6,2,2), non-isomorphic, Weisfeiler-Leman-indistinguishable, and
**cospectral**. Cospectrality means the walk and closed-walk features are
identical at every length (the script prints closed-walk counts `Tr(Aᵗ) =
0, 96, 192, 1536, 7680` for both), so
the walk and reachability covers — and any model built on them — are at chance by
construction.

The fix is the **sieve cover**: instead of paths *into* a node, it inspects the
structure *among* the elements covering it — the subgraph induced on each node's
neighbourhood. That subgraph is two triangles in the Rook graph and a 6-cycle in
Shrikhande (the same C₆-vs-2·C₃ motif as the static demo, now a *local* invariant).

![sieve](results/sieve_acc.png)

`WalkCoverNet` (covers = {walk, reachability}) and the trivial baseline sit at
chance; `SieveNet` — the identical architecture with the sieve cover added to its
basis — reaches 100%. The separation comes entirely from putting the sieve cover
in the basis: the GGNN "design message passing by choosing covers" thesis, made
load-bearing on a pair that defeats WL. (The sieve cover here — neighbourhood
structure — is one principled instantiation of a precomposition-refined cover, not
a verbatim reproduction of the paper's construction, which public sources
under-specify.)

## When time is the signal (temporal demo)

`python run_temporal.py`

The static demo above shows a *model* limitation (1-WL can't read global
reachability). The temporal demo shows something stronger and more fundamental: a
*representation* limitation. Collapse a stream of timestamped events into a static
graph and you discard the ordering — and ordering is the lateral-movement signal.

Here both classes share an **identical static-graph distribution**. Every stream
contains the same lateral-movement chain `f → … → g`, the same background, and an
exfiltration event. Only the *timestamps of the chain* differ:

* **attack:** chain edges occur in increasing time order (within a window), so a
  *time-respecting* path `f → g` exists, and exfiltration follows.
* **benign:** the same edges occur scrambled, so no time-respecting path exists.

Because the static distribution is identical, **any** static model is pinned at
the base rate — not because it is too weak, but because the information is not
there. The fix is a *temporal* cover: time-respecting reachability, computed by a
single time-ordered sweep (the temporal analog of summing powers of the adjacency,
with event order replacing multiplication order). The categorical reading is exact
here — the sieve closure becomes the **causal past** of an event.

![temporal PR-AUC](results/temporal_pr_auc.png)

A plain static GCN and an MLP given the v2 static reachability features both sit
at the ~7% base rate; the same MLP given **temporal** cover features reaches
PR-AUC ~0.83. It is not perfect, and that is honest: the residual false positives
are benign activity that *coincidentally* forms a time-respecting chain (a
scrambled benign ordering is increasing end-to-end about 1/5! of the time) — a
real phenomenon that reachability alone cannot resolve and that would need event
correlation to clean up.

![lead time](results/temporal_lead.png)

Time also unlocks a metric a static classifier cannot even define: how long before
exfiltration the temporal path completes. Every attack whose temporal path
completes is flagged with positive lead time — detected before the data leaves.

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
notebooks/         four-part guided tour (covers -> static -> sieve -> temporal),
                   each importing from src/ with rendered outputs
```

## Limitations and honest scope

This is intentionally a minimal demonstration of one mechanism. It is **not** a
deployable detector. In particular:

* **Structure must be the signal.** The advantage exists precisely because the
  classes are 1-WL-equivalent. When node/edge features already carry the signal
  (ports, byte counts, timing), a plain GCN can match this and the cover
  operator's edge disappears.
* **Bounded for tractability.** Reachability operators trend toward dense,
  ~O(n³) computation. Here `K` is small and graphs are tiny; on real estates
  you would need sparse/ego-net-restricted covers, and that trade-off is exactly
  where the expressivity gain could be eroded.
* **Static and synthetic.** Real telemetry is streaming, time-ordered, typed,
  and heavily class-imbalanced. None of that is modelled here.
* **Robustness untested.** An attacker can pad benign-looking hops; a more
  expressive operator is not automatically a more robust one.

## Possible extensions

* Time-ordered covers (paths respecting causal order) for provenance-graph APT
  detection — the natural second domain for the same idea.
* A real-data sanity check on a small subsample of the LANL authentication
  dataset around labelled red-team events.
* Typed/heterogeneous covers per edge type; sparse operators for scale.

## Related work — where this sits

This PoC lives at the intersection of three literatures that rarely cite each
other. The contribution is not a new state of the art; it is a clean bridge
between them.

**Expressivity beyond 1-WL** is a mature, crowded program. Standard MPNNs are
bounded by the 1-WL test (Xu et al., *GIN*, 2019; Morris et al., *k-GNNs*,
arXiv:1810.02244), and the logic/WL correspondence is now well understood
(Grohe, *The Logic of GNNs*, arXiv:2104.14624). The main families that break the
ceiling are higher-order k-WL networks, subgraph GNNs (Bouritsas et al., *GSN*,
arXiv:2006.09252; Bevilacqua et al., *ESAN*, 2022), and substructure/positional
encodings — all sharing the one tension this repo also faces: **expressivity
trades off against scalability**. Recent 2025–26 work chases cheap expressivity
(e.g. invariant-stratified propagation, Hevapathige et al., arXiv:2603.01388,
KDD 2026). The SRG/CSL/BREC isomorphism benchmarks used by the source
paper are the standard yardsticks here, and GSN already reported strong SRG
results when given domain-chosen substructures.

**Topological & categorical deep learning** is where the *cover* idea belongs:
sheaf neural networks (Hansen–Ghrist; Bodnar et al., *Neural Sheaf Diffusion*,
2022), simplicial/cellular networks, and topological deep learning over
combinatorial complexes (Hajij et al., 2023). The live trend is unification —
*Copresheaf Topological Neural Networks* (arXiv:2505.21251) subsumes GNNs,
attention, sheaf nets, and TNNs under one copresheaf formalism — which is the
broader current the Grothendieck framework swims in.

**GNN-based intrusion / APT detection** is active but driven by *different*
concerns than expressivity — chiefly temporal modelling, false-positive rates,
and reproducibility. Representative systems: MAGIC (USENIX Security 2024, masked
graph representation learning), Kairos (arXiv:2308.05034, a temporal GNN
encoder–decoder over whole-system provenance), Slot (CCS 2025, graph
reinforcement learning), CONTINUUM (arXiv:2501.02981, spatio-temporal), and
JBEIL, which targets the static phases of **lateral movement** specifically. A
2025 reproducibility study (ACM REP '25) documents how brittle several of these
pipelines are in practice. Notably, none of this line frames detection as a
1-WL-expressivity problem — which is exactly the gap this PoC points at.

**Positioning.** The detection community thinks temporally and does not frame its
problems in cover/expressivity terms; the expressivity community rarely touches
authentication or provenance graphs. This repo makes the bridge explicit on a
minimal example: a structural detection task that 1-WL provably cannot solve and
that a reachability cover resolves by construction. The natural next step — a
*time-ordered* cover aligned with the TGN-style systems above — is where the
genuine whitespace lies.

## References

* Paper: *Grothendieck Graph Neural Networks Framework: An Algebraic Platform for
  Crafting Topology-Aware GNNs* — arXiv:2412.08835.
  An ICLR 2026 submission of this work was subsequently withdrawn, so its claims
  are best treated as an unrefereed preprint.
* The `flat` vs `segmented` construction is the security reading of the classic
  C₆ vs 2·C₃ counterexample to 1-WL.

*Built as a portfolio proof of concept. Synthetic data only; no real network
telemetry is used.*
