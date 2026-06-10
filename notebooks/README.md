# Notebooks — a guided tour

Four short, self-contained notebooks that explain what these networks are and how
they apply the Grothendieck *cover* idea, building from intuition to the running
code in `src/`, plus three appendices. Read the first four in order:

1. **[01_covers.ipynb](01_covers.ipynb)** — from neighborhoods to covers. Why
   message passing is bounded by 1-WL, what a cover is, the `Tr` homomorphism
   (subgraph composition = matrix multiplication), and the three covers we use —
   **walk**, **reachability**, and **sieve** — with the category-theory intuition
   (sites & sieves) behind the last.
2. **[02_static_blast_radius.ipynb](02_static_blast_radius.ipynb)** — the
   reachability cover beats 1-WL on a blast-radius task (and a humble component
   count ties it: the *signal*, not the model, is what matters).
3. **[03_sieve_cospectral_srgs.ipynb](03_sieve_cospectral_srgs.ipynb)** — a
   cospectral SRG pair (Rook vs Shrikhande) where the walk/reachability covers
   *provably* fail and only the **sieve cover** separates them.
4. **[04_temporal_causal_past.ipynb](04_temporal_causal_past.ipynb)** — a
   **temporal** cover where the sieve closure becomes the literal **causal past**
   of an event, and time-ordering is the lateral-movement signal.

Appendix:

5. **[05_wl_and_cfi.ipynb](05_wl_and_cfi.ipynb)** — certify the WL level with a small
   oracle, then show where the sieve cover *breaks*: a second 3-WL-indistinguishable
   pair (CFI over K4) the sieve cannot separate, because its win is the injected
   substructure, not general expressivity.
6. **[06_sheaves.ipynb](06_sheaves.ipynb)** — the third word in the title. Builds the
   cellular **sheaf** Laplacian, shows its kernel equals the number of components (the
   blast-radius signal), then shows why a learned sheaf network still cannot use it on
   featureless graphs and lands the `SheafNN` at chance.
7. **[07_lanl_probe.ipynb](07_lanl_probe.ipynb)** — the **real-data** test. Runs the
   actual LANL harness on a synthetic slice (the dataset is gated), explaining the
   novelty baseline, warm-up, and continuous-history split, then reports the committed
   real-data numbers: a pre-registered null where neither the cover nor a trained sheaf
   NN beats novelty.

Each notebook imports the verified functions from `src/`, so it stays in sync with
the command-line demos (`run_experiment.py`, `run_sieve.py`, `run_temporal.py`,
`run_wl.py`).

## Running them

`torch` is built against the NumPy 1.x ABI, so use the pinned environment and a
dedicated kernel:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt ipykernel
.venv/bin/python -m ipykernel install --user --name ggnn-venv --display-name "Python (ggnn venv)"
.venv/bin/jupyter lab        # then select the "Python (ggnn venv)" kernel
```

The notebooks are committed with their outputs and figures already rendered, so
they read top-to-bottom without running anything.
