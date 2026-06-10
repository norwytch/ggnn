# Grothendieck in the Shell: Sheaves, Covers, and Sieves for Lateral-Movement Detection

[![ci](https://github.com/norwytch/ggnn/actions/workflows/ci.yml/badge.svg)](https://github.com/norwytch/ggnn/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](requirements.txt)

![Two access-graph estates that look identical locally; a 1-WL GNN cannot separate them, a reachability cover can](results/hero.png)

This repo is the central station for my obsession with applying category and sheaf theory to machine learning and security. It started as an experiment to try and replicate a paper about cover-based GNNs with synthetic data and has now morphed into a small laboratory for anything involving Grothendieck and security. 

The driving question is whether changing the topology a GNN sees
(via covers, sieves, or sheaves) can help it recover a signal (the blast radius a lateral-movement attacker exploits) a standard message-passing GNN
provably cannot? The immediate follow-up is whether it beats the security practitioner standard for each task (LogReg on counting components, time-respecting reachability, edge novelty). It is tested first
on synthetic graphs built to isolate the gap, then on the LANL ARCS authentication
dataset.

This repo shows that covers and sieves can recover the signal standard GNNs cannot. A sheaf NN cannot: the sheaf Laplacian is the morally right tool, but on featureless graphs the network has nothing to learn from. And none of them beats the simpler tool built for each task. Alas! Beauty sometimes earns its place in production, and sometimes utility wins instead.

## The task

Access graphs represent a network: nodes are hosts or accounts, and a directed edge
`u -> v` means an identity on `u` can authenticate to `v`. We classify two estates that
look identical locally but differ in blast radius.

| class | structure | training | meaning |
|------|-----------|----------|---------|
| `flat` | one directed n-cycle (e.g. a 12-cycle) | sizes vary; tested on an unseen size | every host reaches every other; compromise one, reach all |
| `segmented` | k disjoint directed cycles (e.g. two 6-cycles) | sizes and segment count vary; tested on an unseen size | isolated enclaves; compromise one, reach only its enclave |

A flat estate endangers the whole network if compromised, a segmented one only its
enclave. The two are 1-WL-equivalent by construction, which is why a standard GNN
cannot separate them. See [docs/expressivity-and-covers.md](docs/expressivity-and-covers.md).

## Tests and Findings

| question | finding |
|---|---|
| Can a 1-WL GNN tell a flat estate (compromise one host, reach the whole network) from a segmented one (reach only an enclave)? | No, base rate. But a one-line connected-components count nails it, and so does a reachability cover. The signal, not the architecture. |
| Does the framework reach "beyond 3-WL", as the source paper claims? | Only by hand-injecting the discriminating substructure (the GSN trick); the generic covers stay 2-FWL-bounded. The WL levels are certified by an in-repo oracle, not asserted (`run_wl.py`). |
| Does a more sophisticated architecture, a sheaf neural network, help? | No. Base rate on the synthetic tasks; on featureless graphs its restriction maps have nothing to learn from. |
| On real lateral-movement data (LANL), does any of it beat a no-GNN novelty baseline? | No. A pre-registered null across two red-team windows, for both the cover and a trained sheaf NN. |

The static task in numbers (the first row above):

| model | PR-AUC | recall @ 1% FPR |
|---|---|---|
| GCN, GIN (1-WL) | ~0.03 (= base rate) | 0 |
| SheafNN (2020) | ~0.03 (= base rate) | 0 |
| LogReg on # components | 1.0 | 1.0 |
| GCN + reachability feature | 1.0 | 1.0 |
| CoverNet (cover features) | 1.0 | 1.0 |

~3% positives, tested on a graph size never seen in training. Once reachability is
exposed, logistic regression on one scalar solves it; the sheaf NN sits at the base rate
alongside the plain GNNs.

## Quickstart

```bash
pip install -r requirements.txt
python run_experiment.py     # static: 1-WL GNNs vs the reachability cover (~1 min, CPU)
python run_sieve.py          # sieve cover separates a cospectral SRG pair
python run_temporal.py       # temporal cover: when event ordering is the signal
python run_lanl.py --smoke   # real-data probe harness, on a synthetic slice
python run_wl.py             # certify the Rook/Shrikhande WL level + cover table
```

The demos write figures to `results/`, committed so they render here without running;
`run_wl.py` prints tables instead.

## Demos

| demo | what it shows | deep dive |
|---|---|---|
| static | 1-WL GNNs sit at the base rate; the reachability cover (or a one-line component count) solves it | [expressivity-and-covers](docs/expressivity-and-covers.md) |
| sieve | swapping in the sieve cover separates Rook vs Shrikhande, a cospectral WL-indistinguishable pair | [expressivity-and-covers](docs/expressivity-and-covers.md#the-sieve-cover) |
| temporal | identical static graphs, only event ordering differs; a temporal cover recovers it (PR-AUC ~0.83) | [temporal](docs/temporal.md) |
| LANL probe | the honest test on real data: across two red-team windows neither the cover nor a trained sheaf NN beats a no-GNN novelty baseline (the pre-registered null) | [lanl-probe](docs/lanl-probe.md) |
| WL certification | certifies the Rook/Shrikhande and CFI pairs are 3-WL-indistinguishable, and shows where the sieve cover breaks | [expressivity-and-covers](docs/expressivity-and-covers.md#where-the-sieve-breaks) |

## Repository map

| directory | purpose |
|---|---|
| `src/` | the library: data generators, graph operators, cover algebra, and models. Everything the demos and notebooks import. |
| `docs/` | written deep dives behind each demo. Read these for the why; the README is the map. |
| `notebooks/` | a four-part guided tour (covers, static, sieve, temporal) plus appendices on WL/CFI and sheaves, with rendered outputs, so they read without running. |
| `results/` | committed figures the demos write, embedded in the README and docs. |
| `tests/` | smoke tests asserting the structural invariants the demos depend on. |
| `.github/` | CI: runs the tests and smoke-runs two demos on every push. |

Run scripts live at the root: `run_experiment.py`, `run_sieve.py`, `run_temporal.py`,
`run_lanl.py`, `run_wl.py`. See [docs/repo-structure.md](docs/repo-structure.md) for a
file-level map.

## Notebooks

[`notebooks/`](notebooks/) builds each idea from scratch with inline math and plots.
Start with [01_covers](notebooks/01_covers.ipynb). Kernel setup is in
[notebooks/README.md](notebooks/README.md).

## More

The source framework is from a withdrawn, disputed ICLR 2026 preprint, so treat its
claims as unrefereed (see [References](docs/related-work.md#references)). 

- [Expressivity and covers](docs/expressivity-and-covers.md)
- [When time is the signal](docs/temporal.md)
- [Real-data probe (LANL)](docs/lanl-probe.md)
- [Related work and references](docs/related-work.md)
- [Repository structure](docs/repo-structure.md)
