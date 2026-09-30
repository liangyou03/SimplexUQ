"""Plotting utilities for simplex and coverage figures."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def simplex_triangulation(resolution: int = 50):
    """Create a triangulation grid on the 2-simplex for heatmap plotting."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    pts = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            a = i / resolution
            b = j / resolution
            c = 1.0 - a - b
            pts.append((a, b, c))
    bary = np.asarray(pts)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (np.sqrt(3.0) / 2.0) * bary[:, 2]
    return bary, Triangulation(x, y)


def plot_stratified_coverage(
    results: dict,
    alpha: float,
    strata_labels: list[str] | None = None,
    ax: plt.Axes | None = None,
):
    """Bar plot of stratified coverage across methods."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    methods = list(results)
    strata = sorted({s for vals in results.values() for s in vals})
    labels = strata_labels or [str(s) for s in strata]
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(strata))
    for offset, method in enumerate(methods):
        vals = [results[method].get(s, np.nan) for s in strata]
        ax.bar(x - 0.4 + width / 2 + offset * width, vals, width=width, label=method)
    ax.axhline(1.0 - alpha, color="black", linewidth=1, linestyle="--", label="target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    return ax
