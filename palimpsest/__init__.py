"""
palimpsest — the second movement of ggnn.

ggnn showed that sheaf networks sit at the base rate on featureless tasks, which is
uninformative about sheaves: a sheaf learns its restriction maps from features, and there
were none. palimpsest gives them features. It tests the modern learned-restriction-map
family (Neural Sheaf Diffusion and kin) on featureful heterophilic node classification,
calibrating against the corrected heterophily benchmarks (Platonov et al. 2023) before
moving to fraud graphs, where heterophily is the camouflage signature.

This package is the calibration gate (Act 2): a Platonov loader and the node-level model
matrix. Same oracle-don't-assert, pre-registered-null discipline as the parent.
"""
