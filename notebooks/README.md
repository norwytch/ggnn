# Notebooks — a guided tour

Four short, self-contained notebooks that explain what these networks are and how
they apply the Grothendieck *cover* idea, building from intuition to the running
code in `src/`. Read them in order:

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

Each notebook imports the verified functions from `src/`, so it stays in sync with
the command-line demos (`run_experiment.py`, `run_sieve.py`, `run_temporal.py`).

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
