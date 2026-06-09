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
from src.wl import distinguishes
from src.covers import walk_cover, reachability_cover, sieve_cover


def _separates(fa, fb, atol=1e-6):
    """True if two per-node feature blocks differ as multisets of rows (i.e. the cover
    gives the two graphs distinguishable feature distributions)."""
    a = np.sort(np.round(np.asarray(fa) / atol) * atol, axis=0)
    b = np.sort(np.round(np.asarray(fb) / atol) * atol, axis=0)
    return a.shape != b.shape or not np.allclose(a, b, atol=atol)


def main():
    R, S = rook44(), shrikhande()

    print("\nWL certification: 4x4 Rook's graph vs Shrikhande graph (both SRG(16,6,2,2))\n")
    print(f"{'test':18s} {'distinguishes?':>16s}")
    print("-" * 36)
    for level in ("1-WL", "3-WL", "4-WL"):
        d = distinguishes(R, S, level)
        note = "" if level != "3-WL" else "   <- the bar the claim is about"
        print(f"{level + ' (' + {'1-WL':'1-WL','3-WL':'2-FWL','4-WL':'3-FWL'}[level] + ')':18s} "
              f"{str(d):>16s}{note}")
    print("\nSo the pair is genuinely 3-WL-indistinguishable; only 4-WL separates it.\n")

    covers = {
        "walk (closed walks)":   lambda A: walk_cover(A, 4),
        "reachability":          lambda A: reachability_cover(A, A.shape[0]),
        "sieve (substructure)":  lambda A: sieve_cover(A),
    }
    print(f"{'cover':24s} {'separates pair?':>16s}   kind")
    print("-" * 64)
    for name, f in covers.items():
        sep = _separates(f(R), f(S))
        kind = "GSN substructure (injected)" if "sieve" in name else "framework-generic (2-FWL-bounded)"
        print(f"{name:24s} {str(sep):>16s}   {kind}")

    print("\nReading: generic covers fail on a 3-WL-indistinguishable pair, as a 2-FWL")
    print("bound predicts. The sieve cover clears it, but the power comes from the")
    print("hand-injected triangle count (GSN), not the framework's algebra.\n")


if __name__ == "__main__":
    main()
