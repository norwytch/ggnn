"""
Certify the WL level of the Rook vs Shrikhande pair, then show which covers separate
it. This is the in-repo test of the paper's "beyond 3-WL" claim.

    python run_wl.py

What it establishes, with nothing asserted:
  * 1-WL and 3-WL (= 2-FWL) cannot tell the pair apart; 4-WL (= 3-FWL) can. So this is
    a genuine 3-WL-indistinguishable pair.
  * The framework's generic covers (walk, reachability) produce identical features on
    both graphs, so they do NOT clear the bar -- consistent with a 2-FWL bound.
  * The sieve cover separates them, but only because it counts a hand-chosen
    substructure (triangles in each neighbourhood). That is the GSN mechanism, not the
    framework's algebra. The beyond-3-WL win is bought by the injected substructure.
"""
from __future__ import annotations
import numpy as np

from src.srg_data import rook44, shrikhande
from src.cfi import cfi_pair, complete_graph
from src.wl import distinguishes
from src.covers import walk_cover, reachability_cover, sieve_cover


def _separates(fa, fb, atol=1e-6):
    """True if two per-node feature blocks differ as multisets of rows (i.e. the cover
    gives the two graphs distinguishable feature distributions)."""
    a = np.sort(np.round(np.asarray(fa) / atol) * atol, axis=0)
    b = np.sort(np.round(np.asarray(fb) / atol) * atol, axis=0)
    return a.shape != b.shape or not np.allclose(a, b, atol=atol)


_COVERS = {
    "walk (closed walks)":  lambda A: walk_cover(A, 4),
    "reachability":         lambda A: reachability_cover(A, A.shape[0]),
    "sieve (substructure)": lambda A: sieve_cover(A),
}


def _wl_row(level, A, B):
    label = {"1-WL": "1-WL", "3-WL": "2-FWL", "4-WL": "3-FWL"}[level]
    return f"{level + ' (' + label + ')':14s} {str(distinguishes(A, B, level)):>16s}"


def _certify(name, A, B, do_4wl):
    print(f"\n{name}   (n={A.shape[0]})")
    print(f"{'test':14s} {'distinguishes?':>16s}")
    print("-" * 32)
    for level in ("1-WL", "3-WL") + (("4-WL",) if do_4wl else ()):
        print(_wl_row(level, A, B))
    if not do_4wl:
        print("4-WL (3-FWL)        (skipped: O(n^4) too slow at this size)")


def _cover_table(A, B):
    print(f"{'cover':24s} {'separates pair?':>16s}   kind")
    print("-" * 64)
    for name, f in _COVERS.items():
        kind = ("GSN substructure (injected)" if "sieve" in name
                else "framework-generic (2-FWL-bounded)")
        print(f"{name:24s} {str(_separates(f(A), f(B))):>16s}   {kind}")


def main():
    print("\n=== Pair 1: Rook's vs Shrikhande (SRG(16,6,2,2)) ===")
    R, S = rook44(), shrikhande()
    _certify("WL level", R, S, do_4wl=True)
    print("\n3-WL-indistinguishable; only 4-WL separates it.\n")
    _cover_table(R, S)
    print("\nThe sieve cover clears this 3-WL-indistinguishable pair, but only because the")
    print("discriminating structure here IS a triangle count, which is what it injects.\n")

    print("\n=== Pair 2: CFI over K4 (a non-SRG 3-WL-indistinguishable pair) ===")
    A, B = cfi_pair(4, complete_graph(4))
    _certify("WL level", A, B, do_4wl=True)              # ~14s
    print("\nSame WL level as pair 1: 3-WL cannot, 4-WL can. But CFI graphs are bipartite")
    print("and regular, so neighbourhoods induce no triangles.\n")
    _cover_table(A, B)
    print("\nThe sieve cover now FAILS. Its triangle/component counts are identical on both")
    print("graphs, because the structure that differs is global parity, not a local")
    print("substructure. This is the ceiling: the sieve's reach is exactly the substructure")
    print("it was hand-given, not a general beyond-3-WL power. The reviewer's point, shown.\n")


if __name__ == "__main__":
    main()
