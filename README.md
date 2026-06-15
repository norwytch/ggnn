# Grothendieck in the Shell: Sheaves, Covers, and Sieves for Lateral-Movement Detection

[![ci](https://github.com/norwytch/ggnn/actions/workflows/ci.yml/badge.svg)](https://github.com/norwytch/ggnn/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](requirements.txt)

![Two access-graph estates that look identical locally; a 1-WL GNN cannot separate them, a reachability cover can](results/hero.png)

This started as an attempt to replicate a paper about cover-based graph neural networks and turned into a small laboratory for one stubborn question: whether changing the topology a network sees, through covers, sieves, sheaves, the whole Grothendieck toolbox, can recover a signal a standard model provably cannot, namely the blast radius a lateral-movement attacker exploits.

BLUF: the fancy machinery can recover the signal a plain GNN misses, but it never beats the boring tool a practitioner would already reach for. Not on the synthetic tasks, not on real authentication logs, not even on the featureful data where sheaves are supposed to shine. Beauty sometimes earns its place in production, and sometimes utility wins instead. Further experiments are ongoing. 

## The setup

Draw a network as an access graph: a dot per machine or account, an arrow `u -> v` whenever an identity on `u` can authenticate to `v`. An attacker who lands on one machine wants to move, to hop to the next and the next. How far that spread can reach is the blast radius, and it is the thing worth detecting.

So we build two kinds of estate, on purpose, to be as hard as possible to tell apart:

| class | structure | meaning |
|------|-----------|---------|
| `flat` | one directed cycle through every host | compromise one, reach all; maximum blast radius |
| `segmented` | several disjoint cycles | compromise one, reach only its enclave |

One is a catastrophe, the other is fine, but they are built to look identical locally. Every node has one arrow in, one arrow out, and the same constant feature. A standard message-passing GNN only ever sees local neighborhoods, so it gives both estates the same summary and cannot separate them. That is not a bug. It is the 1-Weisfeiler-Leman bound, the classical limit on what these models can distinguish. The signal that would separate them, how much of the graph each node can reach, is global, and no amount of local gossip recovers it.

## What we tried, and what kept winning

The fix is to feed the model a cover: bundles of paths radiating from each node, which expose reachability, the global signal 1-WL was blind to. Hand a model that, and it solves the task perfectly.

| model | PR-AUC | recall @ 1% FPR |
|---|---|---|
| GCN, GIN (standard, 1-WL) | ~0.03 (= base rate) | 0 |
| SheafNN | ~0.03 (= base rate) | 0 |
| LogReg on # connected components | 1.0 | 1.0 |
| GCN + reachability feature | 1.0 | 1.0 |
| CoverNet (cover features) | 1.0 | 1.0 |

The cover works, but so does a one-line script that counts the connected components. Once the signal is exposed at all, even logistic regression on a single number nails it. The signal was the hard part, not the architecture. This is the refrain, and it recurs.

The rest of the synthetic story is variations on the theme:

- Sieves crack a genuinely hard pair: the cospectral Rook's and Shrikhande graphs, provably indistinguishable up to 3-WL. But they manage it only because we hand-fed them the discriminating substructure, an old trick called GSN. The WL levels are certified by an in-repo oracle (`run_wl.py`), not asserted, and we show exactly where the sieve breaks on a harder CFI pair.
- Sheaves are the most beautiful idea in the building. The sheaf Laplacian's kernel literally equals the component count, the morally correct tool. And they sit at the base rate, because these tasks are featureless by design and a sheaf learns its edge-maps from features. Handed nothing, the right tool does nothing.

Deep dive: [docs/expressivity-and-covers.md](docs/expressivity-and-covers.md). There is also a temporal demo, where the signal is the ordering of events rather than the static graph ([docs/temporal.md](docs/temporal.md)).

## Then we tried it for real

Synthetic tasks are circular, because we chose them, so of course our toys can win. So we ran the LANL authentication dataset, 58 days of real logs with labeled red-team attacks. We pre-registered the prediction first: covers will lose to a boring "is this login new" novelty baseline, because real attackers light up novelty, not graph shape.

On one slice the cover appeared to win. We held that result for about an hour, until we made the model-fitting numerically honest and replicated on a second slice, at which point the win evaporated. It had been a fitting artifact. They do not beat the novelty baseline. The pre-registered null held. See [docs/lanl-probe.md](docs/lanl-probe.md).

Then the obvious objection: sheaves were tested on featureless graphs, which is rigged. Give them the featureful heterophily they were built for. Fair. That became a second movement, palimpsest, with a faithful sheaf network (real learned rotations on every edge) against a well-tuned ordinary GNN, GraphSAGE, on the standard heterophily benchmark:

| model (roman-empire, hardest heterophily) | accuracy |
|---|---|
| plain GCN (oversmooths) | 0.27 |
| GraphSAGE (strong, ordinary baseline) | 0.81 |
| faithful sheaf network | 0.77 |

The sheaf is genuinely good now. It crushes plain GCN, no excuses. And it still loses to the boring tuned baseline. Same story, fair fight, one level up. 

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py        # static: standard GNNs vs the reachability cover (~1 min, CPU)
python run_sieve.py             # sieve cover separates a cospectral SRG pair
python run_temporal.py          # temporal cover: when event ordering is the signal
python run_wl.py                # certify the Rook/Shrikhande WL level + cover table
python run_lanl.py --smoke      # the real-data probe harness, on a synthetic slice
python run_gate.py --synthetic  # palimpsest: GNN vs sheaf on heterophily (synthetic plumbing)
```

Demos write their figures to `results/` (committed, so they render here without running). `run_wl.py` and `run_gate.py` print tables.

## Find your way around

| where | what |
|---|---|
| `src/`, `palimpsest/` | the libraries: data generators, graph operators, cover algebra, and models for the two movements |
| `docs/` | the per-topic deep dives behind each demo |
| `notebooks/` | a guided tour (covers, static, sieve, temporal) plus WL/CFI and sheaf appendices, rendered so they read without running |
| `results/` | committed figures |
| `tests/`, `.github/` | smoke tests and the CI that runs them and every demo on each push |

Docs: [expressivity and covers](docs/expressivity-and-covers.md), [when time is the signal](docs/temporal.md), [real-data probe (LANL)](docs/lanl-probe.md), [related work and references](docs/related-work.md), [repository structure](docs/repo-structure.md).

The source framework is from a withdrawn, disputed ICLR 2026 preprint, so treat its claims as unrefereed (see [References](docs/related-work.md#references)).
