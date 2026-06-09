"""
Render the README hero diagram: two estates that look identical locally, a 1-WL GNN
that cannot separate them, and a cover that can.

    python make_hero.py        # writes results/hero.png and results/hero.svg

A first-pass schematic, drawn as shapes on an axis-off canvas (not a chart), using the
shared figure palette. Meant as a starting point you can refine in Excalidraw.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from src.plotstyle import use_style, BLUE, RED

use_style()

NODE_FILL = "#eef2f6"
NODE_EDGE = "#33414f"
ARROW = "#b3bac2"
INK = "#1f2933"
MUTE = "#7b8794"


def draw_cycle(ax, center, radius, n, start=90, node_r=0.12):
    cx, cy = center
    ang = np.deg2rad(start - np.arange(n) * 360.0 / n)      # clockwise
    pts = np.c_[cx + radius * np.cos(ang), cy + radius * np.sin(ang)]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        u = (b - a) / np.hypot(*(b - a))
        ax.add_patch(FancyArrowPatch(a + u * node_r, b - u * node_r, arrowstyle="-|>",
                                     mutation_scale=8, lw=1.1, color=ARROW, zorder=1))
    for p in pts:
        ax.add_patch(Circle(p, node_r, facecolor=NODE_FILL, edgecolor=NODE_EDGE,
                            lw=1.3, zorder=2))
    return pts


def header(ax, x, y, s):
    ax.text(x, y, s, ha="center", va="center", color=MUTE, fontsize=11,
            fontweight="bold")


def verdict(ax, x, y, ok, size=0.22):
    """Draw a check (ok) or cross as line shapes, to avoid missing font glyphs."""
    color = BLUE if ok else RED
    kw = dict(color=color, lw=3.2, solid_capstyle="round", zorder=3)
    if ok:
        ax.plot([x - size, x - 0.25 * size, x + 1.2 * size],
                [y - 0.1 * size, y - size, y + 1.1 * size], **kw)
    else:
        ax.plot([x - size, x + size], [y - size, y + size], **kw)
        ax.plot([x - size, x + size], [y + size, y - size], **kw)


def main():
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")

    # ---- column 1: two locally-identical estates ----------------------------
    header(ax, 3.0, 6.6, "TWO ESTATES  ·  LOCALLY IDENTICAL")
    draw_cycle(ax, (3.0, 4.9), 1.15, 12)
    ax.text(3.0, 3.35, "flat — one cycle, reach all", ha="center", color=INK, fontsize=12)
    draw_cycle(ax, (2.0, 1.7), 0.78, 6)
    draw_cycle(ax, (4.0, 1.7), 0.78, 6)
    ax.text(3.0, 0.35, "segmented — k enclaves, reach one", ha="center", color=INK, fontsize=12)

    # ---- column 2: the two lenses -------------------------------------------
    header(ax, 9.0, 6.6, "TWO LENSES")
    for y in (4.9, 1.7):
        ax.add_patch(FancyArrowPatch((5.3, y), (6.5, y), arrowstyle="-|>",
                                     mutation_scale=14, lw=2, color=MUTE))
    # 1-WL lens (fails)
    ax.text(6.7, 5.25, "1-WL GNN", ha="left", color=INK, fontsize=13, fontweight="bold")
    ax.text(6.7, 4.7, "both collapse to one embedding", ha="left", color=MUTE, fontsize=11.5)
    verdict(ax, 10.3, 4.95, ok=False)
    # cover lens (works)
    ax.text(6.7, 2.05, "reachability cover", ha="left", color=INK, fontsize=13, fontweight="bold")
    ax.text(6.7, 1.5, "A^t · 1 differs by class", ha="left", color=MUTE, fontsize=11.5)
    verdict(ax, 10.3, 1.75, ok=True)

    # ---- column 3: the payoff -----------------------------------------------
    header(ax, 13.7, 6.6, "BLAST RADIUS")
    x0, x1, yb = 12.1, 15.4, 3.4
    ax.annotate("", xy=(x1 + 0.15, yb), xytext=(x0 - 0.15, yb),
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.4))
    for xt, lab in [(x0, "0"), (x1, "1")]:
        ax.plot([xt, xt], [yb - 0.08, yb + 0.08], color=MUTE, lw=1.4)
        ax.text(xt, yb - 0.4, lab, ha="center", color=MUTE, fontsize=11)
    xs = x0 + (1.0 / 3.0) * (x1 - x0)
    ax.add_patch(Circle((xs, yb), 0.13, color=BLUE, zorder=3))
    ax.text(xs, yb + 0.45, "segmented\n≈ 1/k", ha="center", va="bottom", color=BLUE, fontsize=11)
    ax.add_patch(Circle((x1, yb), 0.13, color=RED, zorder=3))
    ax.text(x1, yb + 0.45, "flat\n≈ 1.0", ha="center", va="bottom", color=RED, fontsize=11)
    verdict(ax, 12.95, 1.78, ok=True, size=0.16)
    ax.text(13.25, 1.7, "separated", ha="left", color=BLUE, fontsize=13, fontweight="bold")

    # ---- caption ------------------------------------------------------------
    ax.text(8.0, -0.15, "Standard GNNs can't separate these estates. A Grothendieck cover can.",
            ha="center", color=INK, fontsize=13.5)

    fig.savefig("results/hero.png", bbox_inches="tight", pad_inches=0.3)
    fig.savefig("results/hero.svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("Wrote results/hero.png and results/hero.svg")


if __name__ == "__main__":
    main()
