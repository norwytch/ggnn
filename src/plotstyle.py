"""
Shared matplotlib styling so the committed figures read clean and consistent rather
than stock-default. Call use_style() once before plotting. Keep titles short here and
let the README/doc prose caption the figure.

Palette: one blue for the result that works, one red for the blind/failing case, gray
for neutral, and a muted tone for reference lines.
"""
from __future__ import annotations
import matplotlib as mpl

BLUE = "#2b6cb0"      # the cover / signal that solves the task
RED = "#c0392b"       # the 1-WL-blind / failing case
GRAY = "#aab2bd"      # neutral bars
REF = "#6b7280"       # dashed reference lines (base rate, chance, exfil)


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.autolayout": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titlepad": 10,
        "axes.labelsize": 12,
        "axes.labelcolor": "#1f2933",
        "axes.edgecolor": "#cbd2d9",
        "text.color": "#1f2933",
        "xtick.color": "#52606d",
        "ytick.color": "#52606d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#eceef0",
        "grid.linewidth": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 11,
    })
