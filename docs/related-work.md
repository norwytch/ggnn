# Related work

This sits at the intersection of three literatures that rarely cite each other. It is
not a new state of the art; it is a bridge between them.

Expressivity beyond 1-WL. Standard MPNNs are bounded by the 1-WL test (Xu et al., GIN,
2019; Morris et al., k-GNNs, arXiv:1810.02244), and the logic/WL correspondence is well
understood (Grohe, arXiv:2104.14624). The families that break the ceiling, higher-order
k-WL networks, subgraph GNNs (Bouritsas et al., GSN, arXiv:2006.09252; Bevilacqua et
al., ESAN, 2022), and substructure encodings, all face the same expressivity-versus-
scalability tension this repo does. SRG/CSL/BREC are the standard benchmarks, and GSN
already reported strong SRG results given chosen substructures, which is why the sieve
demo is best read as re-deriving a known result.

Topological and categorical deep learning is where the cover idea belongs: sheaf neural
networks (Bodnar et al., Neural Sheaf Diffusion, 2022), simplicial and cellular
networks, and topological deep learning over combinatorial complexes (Hajij et al.,
2023). The live trend is unification; Copresheaf Topological Neural Networks
(arXiv:2505.21251) subsume GNNs, attention, sheaf nets, and TNNs under one formalism.

GNN-based intrusion and APT detection is active but driven by other concerns: temporal
modelling, false-positive rates, reproducibility. Representative systems are MAGIC
(USENIX Security 2024), Kairos (arXiv:2308.05034), Slot (CCS 2025), CONTINUUM
(arXiv:2501.02981), and JBEIL, which targets lateral movement. A 2025 reproducibility
study (ACM REP 2025) documents how brittle several pipelines are. None frames detection
as a 1-WL-expressivity problem, the gap this repo points at.

Positioning. The detection community thinks temporally but not in cover/expressivity
terms; the expressivity community rarely touches authentication graphs. This repo makes
the bridge explicit on a minimal example. The natural next step, a time-ordered cover
aligned with the systems above, is where the whitespace lies.

## References

- Paper: Grothendieck Graph Neural Networks Framework: An Algebraic Platform for
  Crafting Topology-Aware GNNs, arXiv:2412.08835. An ICLR 2026 submission of this work
  was withdrawn, so treat its claims as an unrefereed preprint. The OpenReview page
  ([forum 4hXoo5MhxZ](https://openreview.net/forum?id=4hXoo5MhxZ)) shows the status
  "ICLR 2026 Conference Withdrawn Submission" (last modified 2025-11-20). The arXiv
  preprint itself was not retracted; only the conference submission was withdrawn.
- The `flat` vs `segmented` construction is the security reading of the classic C6 vs
  2C3 counterexample to 1-WL.
