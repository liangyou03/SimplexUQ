"""Visualize per-strata coverage from bulk deconvolution experiment."""
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tables/exp2_1_bulk_deconv.json",
                        help="Path to results JSON file")
    parser.add_argument("--output", default="results/figures/fig2_1_bulk_deconv_coverage.pdf",
                        help="Output figure path")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    agg = data["aggregated"]
    methods = list(agg.keys())
    target_coverage = 0.90

    # Extract per-strata coverage for each method
    results = {}
    for method in methods:
        strata_cov = agg[method]["stratified_coverage"]
        strata_list = sorted(int(s) for s in strata_cov.keys())
        coverage_means = [strata_cov[str(s)]["mean"] for s in strata_list]
        coverage_stds = [strata_cov[str(s)]["std"] for s in strata_list]
        results[method] = {
            "strata": strata_list,
            "means": coverage_means,
            "stds": coverage_stds,
        }

    # Create figure: 4 subplots (one per method)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for ax, method in zip(axes, methods):
        r = results[method]
        x = np.arange(len(r["strata"]))
        width = 0.6
        bars = ax.bar(x, r["means"], width, yerr=r["stds"], capsize=5,
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)

        # Add target line
        ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2, label='Target (0.90)')

        ax.set_xlabel('Stratum (0=boundary, 4=center)', fontsize=11)
        ax.set_ylabel('Coverage', fontsize=11)
        ax.set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s}' for s in r["strata"]])
        ax.set_ylim([0.7, 1.05])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle('Bulk Deconvolution: Per-Stratum Coverage (Exp 2.1)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
