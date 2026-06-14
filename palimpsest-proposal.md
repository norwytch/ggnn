# palimpsest — A Fair Test of Sheaf Neural Networks on Featureful Heterophily

*Project proposal (second movement of `ggnn`) — Julia Quinn, June 2026*
*(Working title: a palimpsest is a manuscript scraped and overwritten, the original showing through to the right reader. Fraud is a palimpsest — a benign surface over a true label — and a learned restriction map is the reader that recovers the underlayer. Rename at will.)*

## Summary

`ggnn` reports that a sheaf neural network "sits at the base rate" on the synthetic lateral-movement tasks, and is careful to say why: the tasks are featureless **by construction** — every node carries the same constant feature, which is what makes the two estates 1-WL-equivalent in the first place — and a sheaf network learns its restriction maps *from features*. So that is an honest non-result about *these inputs*, not a verdict on sheaves; it leaves the real question open. This project answers it. Two things compound the confound: the `ggnn` SNN learns only a single graph-wide restriction map (constant, vertex-transitive inputs give a per-edge map nothing to vary on), so it never exercises the heterophily-specialization that is the whole point of the modern family (Bodnar et al. 2022); and the tasks carry no feature signal for any sheaf to use.

This project closes that gap. It (1) upgrades the SNN baseline to the modern learned-restriction-map family (Neural Sheaf Diffusion and its 2022–2025 descendants), (2) tests them on **featureful** graphs where heterophily gives the restriction maps something to specialize on, calibrating first against the corrected heterophily benchmark literature, and (3) lands on **fraud / anti-money-laundering graphs**, where heterophily is not incidental but *intrinsic to the threat model* — fraudsters transact with benign accounts to camouflage — which is exactly the regime Neural Sheaf Diffusion claims to help with, and which keeps the repo's security identity intact.

The deliverable is honest either way. If modern SNNs earn their keep on fraud graphs, `ggnn` gains a second act in which beauty finally wins. If they do not — if well-tuned standard GNNs match them, as Platonov et al. found across the clean heterophily benchmarks — that is the repo's own "utility beats beauty" thesis recurring one level up, pre-registered and clean.

## Motivation — the confound being fixed

Three confounds in the current SheafNN claim, in decreasing order of severity:

1. **Featureless inputs.** Restriction maps are learned from node features; constant features leave nothing to orient against. Any SNN must collapse to base rate here. This is not evidence about sheaves.
2. **A homogeneous model.** The `ggnn` SNN learns one graph-wide restriction map, not the per-edge maps (Bodnar et al. 2022) that are the entire reason sheaves are interesting for heterophily — because featureless, vertex-transitive inputs leave a per-edge map nothing to vary on. `ggnn` does not conclude "sheaves don't help"; it concludes the test was uninformative. The modern learned-restriction-map family is the right thing to run, and it never has been here.
3. **Wrong dataset class.** The synthetic tasks isolate a *structural* signal (blast radius / reachability) with no feature signal. Sheaves are a tool for **feature**-bearing heterophilic node classification. The honest test must move to data of that kind.

## Research questions

Each carries a pre-registered null, scored in the repo's existing finding-table format.

- **RQ1 (calibration).** On the corrected heterophily suite (Platonov et al. 2023), does a learned-restriction-map SNN beat a well-tuned standard GNN (GCN/GraphSAGE + skip connections)?
  *Null:* no — Platonov's own result is that standard GNNs match or beat heterophily-specialized models on clean benchmarks. Reproducing the null first validates the harness; failing to reproduce it means a harness bug, not a discovery.
- **RQ2 (the payoff).** On fraud/AML graphs, where heterophily is the camouflage signature, does an SNN beat (a) the same standard GNNs and (b) the fraud-specialized heterophily baselines (BWGNN, GHRN)?
  *Null:* no improvement over BWGNN at matched recall@1%FPR.
- **RQ3 (mechanism).** Is any SNN gain attributable to the restriction maps specifically, rather than to width/depth/skip-connection budget? Ablate to the trivial (constant) sheaf — which recovers an ordinary GNN — under matched parameter count.
  *Null:* the gain survives the ablation only as noise.
- **RQ4 (expressivity, keeping the oracle spirit).** Two tiers, the second gated on RQ2. *Tier 1 (always, certifiable):* extend `run_wl.py`'s certify-don't-assert discipline to construct graph pairs separable by a learned O(d) restriction map but not by 1-WL, bounding where sheaves add separating power. *Tier 2 (only if RQ2 clears):* audit whether real fraud graphs actually contain that structure — exploratory, because the graph statistic that would measure it is not yet well defined.
  *Null:* the camouflage-edge structure sheaves can exploit is absent from real fraud graphs, and the interpretation is a story, not a mechanism.

## Method

### Model matrix

| family | model | restriction maps | role |
|---|---|---|---|
| simple baseline | LogReg on one scalar where one exists | — | the repo's signature; the floor every model must clear (on fraud: neighborhood-fraud-rate -> LogReg) |
| baseline GNN | GCN, GIN | — (trivial sheaf) | 1-WL floor; the RQ3 ablation target |
| tuned baseline | GraphSAGE / GCN + skip + residual | — | Platonov's "actually fine" standard |
| task baseline | BWGNN, GHRN | — | fraud-specialized heterophily SOTA to beat |
| SNN core | Neural Sheaf Diffusion (Bodnar et al. 2022) | learned diagonal / general / O(d) | the real baseline, not the 2020 one |
| SNN variants | Connection-Laplacian / Sheaf Attention (Barbero et al. 2022) | O(d) via connection | cheaper, often competitive |
| SNN modern | Bundle NN (Bamberger, Barbero et al., arXiv:2405.15540, ICLR 2025) | orthogonal O(d) bundles | efficient; a special case of an SNN |
| SNN recent | Cooperative Sheaf NN (Ribeiro et al., arXiv:2507.00647, 2025) | cooperative; directed in/out Laplacians | strong on heterophily + long-range; the directed framing suits transaction graphs; cite/compare |

The simple baseline (top row) is the repo's signature and is run on every task: whatever a practitioner would reach for before any GNN. It is the first thing a skeptic checks, so it is first-class, not a footnote.

### The interpretive hook

A learned O(d) restriction map on edge `u → v` can encode a reflection or rotation: *information crossing this edge should be inverted before it is believed.* That is a sheaf-theoretic statement of adversarial camouflage — "my neighbor looks benign, so discount or invert their evidence." Whether real fraud graphs reward this (RQ4) is the question; the hook is what makes the security framing more than decoration.

### Expressivity, certified not asserted

Per the repo's existing oracle: do not claim "sheaves separate X" without an in-repo check. Reuse the WL-level certifier to bound where learned restriction maps add separating power over 1-WL, and audit whether the fraud datasets contain the separable structure. A negative here is as publishable as a positive and more honest than most of the field.

## Experimental design — three acts

- **Act 1 (recap, existing).** The featureless structural tasks. SNNs at base rate — re-stated, now correctly framed as *uninformative about sheaves* rather than as evidence against them. One paragraph; it sets up the confound.
- **Act 2 (calibration, new).** Platonov's five corrected heterophilic datasets. Reproduce the field: standard GNNs strong, SNNs roughly matched. If the harness cannot reproduce this, stop and debug — everything downstream is invalid otherwise. **This is the gate, and it is also the MVP:** a clean calibration plus the RQ3 ablation on the Platonov suite is a complete, shippable study on its own, so the contribution does not depend on Act 3 clearing RQ2.
- **Act 3 (payoff, new).** Fraud/AML graphs via GADBench's standardized splits. The real test of whether sheaves earn their place when heterophily *is* the threat signal. Pre-registered nulls (RQ2) fixed before the runs.

Metrics: AUC-PR and recall@1%FPR throughout (severe class imbalance — accuracy is meaningless); AUROC additionally for the binary sets — the Platonov minesweeper/tolokers/questions sets *and* the binary fraud sets — to match leaderboard conventions. **Elliptic uses time-respecting splits** or it leaks. Splits, seeds, and nulls registered in a `preregistration.md` before any Act 3 run, mirroring the LANL pre-registered null already in the repo.

## Datasets

| dataset | domain | nodes | features | homophily | why |
|---|---|---|---|---|---|
| roman-empire | word-dependency | 22.7k | 300 | 0.05 | strongest heterophily; calibration |
| amazon-ratings | co-purchase | 24.5k | 300 | 0.38 | multiclass calibration |
| minesweeper / tolokers / questions | synthetic / crowd / QA | — | yes | binary AUROC calibration |
| YelpChi + Amazon (fraud) | review fraud | — | yes | the standard fraud pair |
| T-Finance / T-Social | finance / social | — | yes | larger, real |
| Elliptic | Bitcoin AML | 203k | 166 | — | temporal; ties to the temporal-cover work |
| DGraph-Fin | financial | ~3M | yes | scale stress test |

Pull fraud datasets through **GADBench** rather than reassembling loaders — it standardizes splits and metrics and gives a leaderboard to position against. Skip the discredited Squirrel/Chameleon entirely (duplicate-node leakage); if used at all, only the Platonov-filtered versions.

## Milestones with kill criteria

- **M0 — harness (wk 1).** Wire NSD into the existing model registry next to GCN/GIN. *Kill:* if NSD cannot be made to run on one Platonov dataset within a week, the integration cost outweighs the contribution — revert to a baselines-only study.
- **M1 — the gate, highest-information experiment (wk 2).** Single plot: NSD vs. GCN+skip on roman-empire. *Kill:* if NSD underperforms a *plain untuned* GCN here, the harness is wrong (NSD should at least match on the strongest-heterophily set); fix before proceeding. Spend the first free weekend here.
- **M2 — Act 2 complete (wk 3–4).** Full model matrix × 5 Platonov datasets. *Gate:* harness reproduces the literature's "standard GNNs competitive" result within CI. *Kill:* if it does not reproduce after debugging, the study is not trustworthy enough to extend to fraud — write up the calibration failure honestly and stop.
- **M3 — Act 3, fraud (wk 5–6).** Model matrix × GADBench fraud set, nulls pre-registered. RQ2 + RQ3 ablation. *Kill:* none — a clean null is the result.
- **M4 — expressivity audit (wk 6–7).** RQ4 tier 1: certify the synthetic separable-vs-1-WL pairs (always). Tier 2 (only if RQ2 cleared): audit their presence in the fraud graphs. *Kill:* none.
- **M5 — writeup (wk 7–9).** Update `ggnn` README to the three-act arc; new `docs/sheaves-on-fraud.md`; arXiv note / workshop submission (TAG-ML, GLFrontiers) if Act 3 cleared RQ2. 4–8 pages, declared limitations in the usual style, regardless of outcome.

## Compute & cost

Acts 1–2 and all fraud sets except DGraph-Fin fit a single consumer GPU. General restriction maps are **O(d²) per edge**; budget this up front rather than discovering it on DGraph-Fin (~3M nodes), where it may force diagonal or O(d) bundle maps — which is itself a finding worth stating (if only the cheap restriction maps scale, the expressive ones are academic at production size).

## Novelty / prior-art check (do before claiming anything)

Sheaves + heterophily is well-trodden (NSD, CSNN, the 2025 "deep geometry to deep learning" survey). Fraud + heterophily is well-trodden (BWGNN, GHRN, SEC-GFD). **Sheaves + fraud** did not surface in a first search and may be the genuine gap — but verify against the GADBench leaderboard, CSNN's related-work, and the BWGNN/GHRN citation graph before asserting it. If someone has done it, the contribution downgrades from "first" to "honest replication with pre-registered nulls," which is still worth shipping.

## Limitations (declared)

- **The Act 2 null may simply repeat.** Platonov's central finding is that tuned standard GNNs are hard to beat on clean heterophily. Sheaves may not clearly win even on fair data — a negative result, consistent with the repo's thesis, not a failure of the study.
- **Fraud heterophily may be the "wrong kind."** Heterophily that helps sheaves is structured (a consistent relabeling captured by restriction maps); fraud heterophily may be closer to noise, in which case no restriction map helps. RQ4 is meant to detect exactly this.
- **Cost may neuter the test.** If only diagonal/O(d) maps scale to the large graphs, the most expressive sheaves never get a fair run at production scale.
- **The camouflage-edge interpretation is a story until RQ4 closes it.** Do not let the prose imply a mechanism the certification has not established.

## Relationship to the portfolio

- **`ggnn` (parent and host).** palimpsest is not a separate repo but the next layer of `ggnn` itself, written over the original featureless-task work and meant to gradually replace it as the project evolves — the palimpsest, with the earlier text still legible underneath. Same oracle-don't-assert and pre-registered-null discipline.
- **`fascicle` (sibling).** Shares the cellular-sheaf substrate; that project uses sheaf *energy* for OOD, this one uses sheaf *diffusion* for classification — the two together cover both faces of the sheaf Laplacian (its kernel and its action).
- **`pitwaller` (reused machinery).** Its conformal / risk-controlled thresholding and CI-gated drift signals port directly into the pre-registered nulls and the recall@FPR operating points here.
