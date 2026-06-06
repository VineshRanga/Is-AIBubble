"""
src/utils/plotting.py
---------------------
Global matplotlib/seaborn style configuration for publication-quality figures
compatible with LaTeX / arXiv submission.

Usage
-----
    from src.utils.plotting import set_paper_style, TICKER_COLORS, SCENARIO_COLORS
    set_paper_style()          # call once at the top of every notebook
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


# ── Publication colour palettes ──────────────────────────────────────────────

TICKER_COLORS: dict[str, str] = {
    "MSFT":  "#0078D4",   # Microsoft blue
    "GOOGL": "#C5221F",   # Google red
    "AMZN":  "#E47911",   # Amazon orange
    "META":  "#1877F2",   # Meta blue
    "AAPL":  "#555555",   # Apple grey
    "NVDA":  "#76B900",   # Nvidia green
    "TSLA":  "#CC0000",   # Tesla red
}

SCENARIO_COLORS: dict[str, str] = {
    "fizzle":   "#2E7D32",   # dark green
    "systemic": "#B71C1C",   # dark red
    "dotcom":   "#E65100",   # deep orange
    "baseline": "#1565C0",   # dark blue
}

INDEX_COLORS: dict[str, str] = {
    "NDX":  "#1565C0",
    "GSPC": "#6A1B9A",
}


# ── Core style function ───────────────────────────────────────────────────────

def set_paper_style(
    font_scale: float = 1.1,
    dpi: int = 300,
) -> None:
    """
    Configure matplotlib and seaborn for publication-quality white-background figures.

    Sets pure white backgrounds, high-contrast black/dark-grey text and axes,
    clean gridlines, and publication-ready DPI. Call once at the top of every
    notebook or script that produces figures for the arXiv paper.

    Parameters
    ----------
    font_scale : float
        Seaborn font scale multiplier. Default 1.1.
    dpi : int
        Figure DPI for both rendering and saving. Default 300.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=font_scale,
        rc={
            "figure.facecolor":    "white",
            "axes.facecolor":      "white",
            "axes.edgecolor":      "#333333",
            "axes.labelcolor":     "#1a1a1a",
            "axes.titlecolor":     "#1a1a1a",
            "xtick.color":         "#333333",
            "ytick.color":         "#333333",
            "text.color":          "#1a1a1a",
            "grid.color":          "#e0e0e0",
            "grid.linewidth":      0.7,
            "legend.facecolor":    "white",
            "legend.edgecolor":    "#cccccc",
            "legend.framealpha":   0.9,
            "figure.dpi":          dpi,
            "savefig.dpi":         dpi,
            "savefig.facecolor":   "white",
            "savefig.bbox":        "tight",
            "font.family":         "DejaVu Sans",
            "axes.spines.top":     False,
            "axes.spines.right":   False,
        },
    )
    plt.rcParams["figure.dpi"]      = dpi
    plt.rcParams["savefig.dpi"]     = dpi
    plt.rcParams["savefig.facecolor"] = "white"
