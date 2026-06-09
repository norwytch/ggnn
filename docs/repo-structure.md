# Repository structure

How the pieces fit. The flow is the same in every demo: `src/data*` builds graphs,
`src/operators` and `src/covers` turn them into features, `src/*models` classify, and a
root `run_*.py` ties it together and writes a figure to `results/`.

## Directories

| directory | what it holds | role |
|---|---|---|
| `src/` | library code: data generators, operators, cover algebra, models | imported by every demo and notebook; nothing here runs on its own |
| `docs/` | prose deep dives behind each demo | the why; the README is the map |
| `notebooks/` | four-part tour with rendered outputs | the same ideas built from scratch, readable without running |
| `results/` | committed PNG figures the demos write | embedded in README and docs |
| `tests/` | pytest smoke tests | assert the structural invariants the demos rely on |
| `.github/` | CI workflow | runs tests and smoke-runs two demos on push |

## `src/` files

Static demo:

- `data.py`: access-graph generator. Flat vs segmented cycles, class imbalance,
  variable size, and a test split on an unseen size.
- `operators.py`: graph primitives. 1-hop propagators, connected components,
  reachable-set counts, blast radius.
- `models.py`: GCN and GIN (1-WL), GCNPlus (GCN plus the reachability feature), and
  CoverNet (an MLP over cover features).

Sieve demo:

- `covers.py`: the cover algebra. The `Tr` homomorphism and the walk, reachability,
  and sieve covers as per-node feature blocks.
- `srg_data.py`: the Rook's vs Shrikhande SRG pair, cospectral and
  WL-indistinguishable.
- `sieve_models.py`: models that combine covers with learnable per-cover gates.
- `wl.py`: a Weisfeiler-Leman oracle (1-WL, 2-FWL, 3-FWL) to certify the WL level at
  which a graph pair becomes distinguishable.
- `cfi.py`: Cai-Fuerer-Immerman graph-pair generator, for a non-SRG
  3-WL-indistinguishable pair where the sieve cover provably breaks.

Temporal demo:

- `temporal_data.py`: event-stream generator where only timestamps separate attack
  from benign.
- `temporal_ops.py`: time-respecting reachability and the feature blocks built on it.
- `temporal_models.py`: static baselines vs the temporal-cover model.

LANL probe:

- `lanl.py`: loader for the real dataset (auth and redteam parsing, time-windowed
  graphs) plus a synthetic slice for a no-download smoke test.
- `baselines.py`: the novelty-plus-degree baseline and the bounded reachability cover,
  as per-edge features.

## Run scripts (root)

- `run_experiment.py`: static demo, with the LogReg-on-components baseline. Writes
  `pr_auc.png` and `separation.png`.
- `run_sieve.py`: sieve demo. Writes `sieve_acc.png`.
- `run_wl.py`: certifies the Rook/Shrikhande WL level and tabulates which covers
  separate it. No figure; prints two tables.
- `run_temporal.py`: temporal demo plus a lead-time metric. Writes `temporal_*.png`.
- `run_lanl.py`: real-data probe. Writes `lanl_pr.png`.
