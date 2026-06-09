# Expressivity and covers

Why a standard GNN fails on `flat` vs `segmented`, what a cover is, how the sieve cover
separates graphs that walks and reachability cannot, and what any of this genuinely
adds over simpler tools.

## Why standard GNNs fail

Message-passing GNNs are bounded by the 1-Weisfeiler-Leman (1-WL) test. In both classes
every node has in-degree 1, out-degree 1, and the same feature, so 1-WL gives every node
the same colour forever and the two classes get the same embedding. GCN and GIN
therefore score at the base rate.

The missing signal is global reachability: the fraction of the estate each node can
reach (about 1.0 when flat, about 1/k when split into k enclaves), equivalently the
number of connected components. This is a property of the whole graph, not any
neighbourhood, so a 1-WL model cannot recover it.

![separation](../results/separation.png)

## What a cover is

A cover generalises a node's neighbourhood to a family of directed paths written as
matrices, where matrix multiplication is path concatenation. `CoverNet` uses the
reachability cover up to `K` hops, exposing per node:

- arriving walks of length `t` (`A^t . 1`),
- closed walks of length `t` (`diag(A^t)`), the cycle signal 1-WL misses,
- the size of the reachable set within `K` hops, the blast radius.

An MLP over these solves the task exactly.

The paper claims the idea scales, with cover/sieve operators separating graphs that
defeat even 3-WL. A reviewer disputed it, arguing the framework is bounded by 2-FWL
(which equals 3-WL). Both can be read together once you separate the framework's generic
covers from the sieve cover, and the sieve demo below settles it on a certified pair
(`run_wl.py`): the generic covers stay 2-FWL-bounded, while the sieve cover clears
beyond-3-WL only by counting a substructure chosen by hand.

## The sieve cover

```bash
python run_sieve.py
```

The reachability cover does no work on the static task that a component count couldn't,
so this demo picks a task where ordinary covers provably fail: distinguishing the 4x4
Rook's graph from the Shrikhande graph. Both are SRG(16,6,2,2), non-isomorphic, and a
standard 3-WL-indistinguishable pair: `run_wl.py` certifies that 1-WL and 3-WL (2-FWL)
cannot tell them apart and only 4-WL (3-FWL) can. They are also cospectral, so their
walk and closed-walk counts agree at every length. The script prints
`Tr(A^t) = 0, 96, 192, 1536, 7680` for both, so any model built on walks or reachability
is at chance.

The walk and reachability covers look outward, along the paths leaving a node. On a
regular graph those counts are constant across nodes and identical between cospectral
graphs, which is why they cannot tell Rook from Shrikhande.

The sieve cover looks sideways instead, at the subgraph a node's neighbours induce among
themselves. It restricts `A` to the neighbour set and counts triangles, components, and
short closed walks inside it. In Rook that subgraph is two triangles; in Shrikhande it is
a single 6-cycle. Same C6-vs-2C3 motif as the static task, now local rather than global.
Both are still covers under the same `Tr` homomorphism; the sieve cover is just a more
refined choice.

![sieve](../results/sieve_acc.png)

`WalkCoverNet` and the trivial baseline sit at chance; `SieveNet`, the same architecture
plus the sieve cover, reaches 100%. The gap is entirely the cover choice.

What this separation is: counting hand-chosen substructures in each neighbourhood.
Separating cospectral SRGs this way is the known Graph Substructure Network (GSN) result,
and a reviewer made exactly this point about the paper. So the win is real but not new
expressivity. What it shows cleanly is the GGNN move: swap one cover for another, hold
the architecture fixed, and the model flips from chance to 100%.

The sieve cover here is one principled instantiation of a precomposition-refined cover,
not a verbatim copy of the paper's construction, which public sources under-specify.

## Where the sieve breaks

The sieve win on Rook/Shrikhande raises a fair question: is the sieve cover a general
beyond-3-WL separator, or did it just happen to look at the substructure that differs?
`run_wl.py` settles it with a second pair at the same WL level, built from a different
mechanism: a Cai-Fuerer-Immerman (CFI) construction over K4 (`src/cfi.py`). The oracle
certifies the CFI pair is also 3-WL-indistinguishable and 4-WL-distinguishable, exactly
like Rook/Shrikhande.

On this pair the sieve cover fails. CFI graphs are bipartite and regular, so every
node's neighbourhood induces no triangles and the sieve's hand-chosen substructure
counts come out identical on both graphs. The structure that actually differs is a
global parity, which no fixed neighbourhood-substructure count can see. So the sieve
cover is not a general beyond-3-WL separator. Its reach is exactly the substructure it
was given, and it clears a pair only when that pair's difference happens to be that
substructure. That is the reviewer's point, made concrete: the expressivity comes from
the injected substructure, not the framework.

## What this genuinely adds

Every task in this repo is tied by something simpler. Connected components solve `flat`
vs `segmented`. Neighbourhood triangle counts (GSN, 2020) solve Rook vs Shrikhande. A
time-respecting reachability check solves the temporal task. Edge novelty is the bar on
real LANL data. In each case the cover feature is something a classical algorithm
computes directly, and once you have it even logistic regression finishes the job. The
GNN machinery never does unique work in these demos.

The framework's one real differentiator is unification under a single learnable object:
express component counts, triangle counts, and temporal reachability all as covers, drop
them into one differentiable model, and let it learn which cover matters (the per-cover
gates in `src/sieve_models.py`). A pile of classical tools cannot be trained end-to-end
with the downstream objective; a cover-based model can.

That only pays off when you do not already know which structural signal matters, the
signal is not a single off-the-shelf statistic, and there is enough labelled data to
learn the gate. The demos violate the first condition by construction, since the task is
chosen, which is why a one-liner ties each one.

Honest positioning: a clear demonstration of a known expressivity gap and a bridge
between literatures, not evidence that cover-based GNNs beat existing tools. The bet is
learnability over hand-engineering, and on this evidence it is unproven. The
[LANL probe](lanl-probe.md) is the first test where the signal is not pre-rigged.
