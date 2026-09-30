"""Plot PBMC sensitivity analyses for appendix use."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS = ["global", "partition", "twostage", "fullcp"]
METHOD_LABELS = {
    "global": "Global",
    "partition": "Mondrian",
    "twostage": "TwoStage",
    "fullcp": "FullCP",
}

SETTING_GROUPS = {
    "Stratification": [
        ("Boundary", "results/tables/pbmc_sensitivity_exp2_1_bulk_deconv_boundary_fixed.json"),
        ("Entropy", "results/tables/pbmc_sensitivity_exp2_1_bulk_deconv_entropy_fixed.json"),
        ("KMeans", "results/tables/pbmc_sensitivity_exp2_1_bulk_deconv_kmeans_fixed.json"),
    ],
    "Mixture concentration": [
        ("0.5", "results/tables/pbmc_concentration_sparse.json"),
        ("1.0", ["results/tables/real_bulk_deconv.json", "results/tables/real_bulk_deconv_fullcp.json"]),
        ("2.0", "results/tables/pbmc_concentration_balanced.json"),
    ],
}

LINE_COLORS = {
    "Boundary": "#0072B2",
    "Entropy": "#D55E00",
    "KMeans": "#009E73",
    "0.5": "#CC79A7",
    "1.0": "#0072B2",
    "2.0": "#E69F00",
}

LINE_MARKERS = {
    "Boundary": "o",
    "Entropy": "s",
    "KMeans": "D",
    "0.5": "o",
    "1.0": "s",
    "2.0": "D",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG_DIR = REPO_ROOT / "paper" / "rewrite_2026" / "latex" / "figures"


def load_summary(path_or_paths: str | list[str]) -> dict:
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]

    merged = {}
    for path in path_or_paths:
        with open(path) as f:
            data = json.load(f)
        summary = data["aggregated"] if "aggregated" in data else data["summary"]
        merged.update(summary)
    return merged


def extract_metric(summary: dict, method: str, metric: str) -> tuple[float, float]:
    entry = summary[method][metric]
    return float(entry["mean"]), float(entry["std"])


def plot_panel(ax, title: str, settings: list[tuple[str, str]]):
    x = np.arange(len(METHODS))
    offsets = np.linspace(-0.22, 0.22, num=len(settings))

    for offset, (label, path) in zip(offsets, settings):
        summary = load_summary(path)
        y = []
        yerr = []
        for method in METHODS:
            mean, std = extract_metric(summary, method, "max_disparity")
            y.append(mean)
            yerr.append(std)
        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            color=LINE_COLORS[label],
            marker=LINE_MARKERS[label],
            markersize=5.5,
            linewidth=1.6,
            elinewidth=1.0,
            capsize=2.5,
            label=label,
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Max disparity")
    ax.set_ylim(0.0, 0.55)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)


def save_figure(fig: plt.Figure, output: Path) -> None:
    """Save the appendix figure to both the results tree and the paper tree."""
    output.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    mirror = PAPER_FIG_DIR / output.name
    fig.savefig(mirror)
    print(f"Saved {output}")
    print(f"Mirrored {mirror}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/figures/fig8_pbmc_sensitivity.pdf",
        help="Output figure path",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    for ax, (title, settings) in zip(axes, SETTING_GROUPS.items()):
        plot_panel(ax, title, settings)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))

    out = Path(args.output)
    save_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
