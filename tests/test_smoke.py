"""
Smoke tests: fast invariants for the data, operators, covers, the SRG pair, and the
LANL probe harness. These don't check model accuracy (that's what the run_*.py demos
report) -- they check that the structural signals the demos rely on actually hold, so
a refactor that quietly breaks one fails here.

    pytest -q
"""
import numpy as np
import pytest

from src import data, operators, covers, srg_data
from src.temporal_ops import temporal_arrivals


# ---- data generator -----------------------------------------------------------

def test_flat_is_one_component_segmented_is_k():
    assert operators.n_components(data._flat(12)) == 1
    assert operators.n_components(data._segmented(12, 2)) == 2
    assert operators.n_components(data._segmented(12, 3)) == 3


def test_blast_radius_separates_classes():
    flat = operators.blast_radius(data._flat(12))
    seg = operators.blast_radius(data._segmented(12, 3))
    # flat: every node reaches ~the whole estate; segmented k=3: ~1/3 each
    assert flat.min() > 0.9
    assert seg.max() < 0.4


def test_make_dataset_tests_on_unseen_size():
    tr_As, tr_y, te_As, te_y = data.make_dataset(seed=0)
    train_sizes = {A.shape[0] for A in tr_As}
    test_sizes = {A.shape[0] for A in te_As}
    assert test_sizes.isdisjoint(train_sizes)         # generalization probe holds
    assert 0 < tr_y.mean() < 0.2                        # imbalanced, some positives


# ---- operators / covers -------------------------------------------------------

def test_reachable_counts_on_a_cycle():
    A = data._flat(8)
    rc = operators.reachable_counts(A, K=None)          # to convergence
    assert np.allclose(rc, 8)                            # a directed cycle reaches all


def test_tr_is_a_monoid_homomorphism():
    assert covers.check_homomorphism()


def test_cover_feature_shapes():
    A = data._flat(10)
    assert covers.walk_cover(A, 4).shape == (10, 8)      # 2 feats * K
    assert covers.reachability_cover(A, 10).shape == (10, 1)
    assert covers.sieve_cover(A, L=6).shape == (10, 6)   # tri, comps, cw len 3..6


# ---- the WL-defeating SRG pair -------------------------------------------------

def test_srg_pair_is_cospectral_and_6_regular():
    R, S = srg_data.rook44(), srg_data.shrikhande()
    assert np.allclose(R.sum(1), 6) and np.allclose(S.sum(1), 6)
    er = np.sort(np.linalg.eigvalsh(R))
    es = np.sort(np.linalg.eigvalsh(S))
    assert np.allclose(er, es, atol=1e-6)                # cospectral
    assert not np.array_equal(R, S)


def test_sieve_separates_what_walks_cannot():
    R, S = srg_data.rook44(), srg_data.shrikhande()
    # walk traces (closed-walk counts) agree -> walk cover is blind
    for t in (2, 3, 4):
        assert np.isclose(np.trace(np.linalg.matrix_power(R, t)),
                          np.trace(np.linalg.matrix_power(S, t)))
    # sieve cover sees the neighbourhood arrangement: triangles in Rook, none in
    # Shrikhande (its neighbourhood is a 6-cycle)
    tri_R = covers.sieve_cover(R)[:, 0]
    tri_S = covers.sieve_cover(S)[:, 0]
    assert tri_R.mean() > 0.5
    assert np.allclose(tri_S, 0.0)


# ---- temporal cover ------------------------------------------------------------

def test_time_respecting_path_depends_on_order():
    # chain a -> b -> c; increasing timestamps connect a..c, scrambled do not
    inc = [("a", "b", 1), ("b", "c", 2)]
    dec = [("a", "b", 2), ("b", "c", 1)]
    assert "c" in temporal_arrivals(inc, src="a")
    assert "c" not in temporal_arrivals(dec, src="a")


# ---- LANL probe harness --------------------------------------------------------

def test_lanl_smoke_pipeline_aligns_and_scores():
    from src.lanl import build_windows, synthetic_slice
    from src import baselines

    auths, red, ws = synthetic_slice(seed=0)
    windows = list(build_windows(iter(auths), ws))
    assert len(windows) > 0

    history = baselines.History()
    total_edges = 0
    saw_positive = False
    for w in windows:
        Xb = baselines.baseline_features(w, history)
        Xc = baselines.cover_features(w, K=3)
        y = baselines.edge_labels(w, red)
        # every block has one row per edge, and the widths match the declared names
        assert Xb.shape == (len(w.edges), len(baselines.BASELINE_NAMES))
        assert Xc.shape == (len(w.edges), len(baselines.COVER_NAMES))
        assert y.shape == (len(w.edges),)
        assert np.isfinite(Xb).all() and np.isfinite(Xc).all()
        saw_positive |= bool(y.sum())
        total_edges += len(w.edges)
        for a in w.edges:
            history.update(a)
    assert total_edges > 0
    assert saw_positive            # the synthetic slice plants some red-team edges


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
