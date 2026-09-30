"""Generate SimplexUQ benchmark figures from saved results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

METHOD_LABELS = {
    "global": "Global",
    "partition": "Mondrian",
    "twostage": "TwoStage",
    "fullcp": "FullCP",
    "jackknife_plus": "Jackknife+",
    "weighted": "Weighted",
    "oracle": "Oracle",
    "oneshot": "OneShot",
    "trainres": "TrainRes",
}

METHOD_COLORS = {
    "global": "#D55E00",
    "partition": "#0072B2",
    "twostage": "#009E73",
    "fullcp": "#56B4E9",
    "jackknife_plus": "#CC79A7",
    "weighted": "#F0E442",
    "oracle": "#000000",
    "oneshot": "#7F7F7F",
    "trainres": "#E69F00",
}

METHOD_ORDER = [
    "global",
    "partition",
    "twostage",
    "fullcp",
    "jackknife_plus",
    "oneshot",
    "trainres",
    "weighted",
    "oracle",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG_DIR = REPO_ROOT / "paper" / "rewrite_2026" / "latex" / "figures"

DGP_SPECS = [
    ("d1_homogeneous", "D1\nHom."),
    ("d2_pure_scale", "D2\nScale"),
    ("d3_discrete_groups_aligned", "D3\nDiscrete"),
    ("d4_model_bias", "D4\nBias"),
    ("d5_heavy_tail", "D5\nTail"),
    ("d6_high_k", "D6$^{\\dagger}$\nHigh-K"),
]

SYNTH_EXTRA_FILES = {
    "d1_homogeneous": ["d1_homogeneous_exact.json"],
    "d3_discrete_groups_aligned": ["d3_discrete_groups_aux.json"],
    "d5_heavy_tail": ["d5_heavy_tail_aux.json"],
    "d6_high_k": ["d6_high_k_aux.json", "d6_high_k_exact_appendix.json"],
}

REAL_SPECS = [
    ("exp2_2_softmax_cifar10_strata_entropy_fixed.json", "CIFAR-10"),
    ("exp2_3_hyperspectral_samson_nmf_all_methods.json", "Samson"),
    ("exp2_5_topics_K10_all_methods.json", "Topics"),
    ("exp2_6_affective_text.json", "AffectiveText"),
    ("exp2_4_age_ldl_K10_image_knn_main.json", "UTKFace"),
    ("real_bulk_deconv.json", "PBMC"),
]

REAL_EXTRA_FILES = {
    "PBMC": [
        "real_bulk_deconv_fullcp.json",
        "real_bulk_deconv_aux.json",
        "real_bulk_deconv_trainres.json",
    ],
    "UTKFace": ["exp2_4_age_ldl_K10_image_knn_fullcp_2k.json"],
}

REAL_MARKERS = {
    "global": "o",
    "partition": "s",
    "twostage": "^",
    "fullcp": "D",
    "jackknife_plus": "P",
    "trainres": "X",
}

PROFILE_MARKERS = {
    "global": "o",
    "partition": "s",
    "twostage": "^",
    "fullcp": "D",
    "jackknife_plus": "P",
    "oracle": "X",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save a figure to the results directory and mirror it into the paper tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    mirror = PAPER_FIG_DIR / filename
    fig.savefig(out)
    fig.savefig(mirror)
    print(f"Saved {out}")
    print(f"Mirrored {mirror}")


def simplex_to_xy(U: np.ndarray) -> np.ndarray:
    """Map 3-simplex points to 2D barycentric coordinates."""
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    return U @ vertices


def extract_summary(data: dict) -> dict:
    if "summary" in data:
        return data["summary"]
    if "aggregated" in data:
        return data["aggregated"]
    raise KeyError("Result file must contain 'summary' or 'aggregated'")


def metric_mean(summary: dict, method: str, metric: str) -> float:
    return float(summary[method][metric]["mean"])


def metric_std(summary: dict, method: str, metric: str) -> float:
    return float(summary[method][metric]["std"])


def available_methods(summary: dict) -> list[str]:
    return [m for m in METHOD_ORDER if m in summary]


def highlight_best_cells(ax, matrix: np.ndarray, methods: list[str], exclude: set[str] | None = None):
    exclude = exclude or set()
    for col in range(matrix.shape[1]):
        best_row = None
        best_val = None
        for row, method in enumerate(methods):
            val = matrix[row, col]
            if method in exclude or np.isnan(val):
                continue
            if best_val is None or val < best_val:
                best_val = val
                best_row = row
        if best_row is None:
            continue
        ax.add_patch(
            Rectangle(
                (col - 0.5, best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )


def highlight_extreme_cells(
    ax,
    matrix: np.ndarray,
    methods: list[str],
    *,
    mode: str = "min",
    exclude: set[str] | None = None,
):
    """Mark the column-wise best cell, using either the minimum or maximum value."""
    exclude = exclude or set()
    for col in range(matrix.shape[1]):
        best_row = None
        best_val = None
        for row, method in enumerate(methods):
            val = matrix[row, col]
            if method in exclude or np.isnan(val):
                continue
            if best_val is None:
                best_val = val
                best_row = row
            elif mode == "min" and val < best_val:
                best_val = val
                best_row = row
            elif mode == "max" and val > best_val:
                best_val = val
                best_row = row
        if best_row is None:
            continue
        ax.add_patch(
            Rectangle(
                (col - 0.5, best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )


def draw_metric_heatmap(
    fig: plt.Figure,
    ax,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    best_mode: str | None,
    row_keys: list[str],
    exclude_best: set[str] | None = None,
):
    """Draw a tickless, data-scaled heatmap with numeric cell labels."""
    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        normalized = matrix.copy()
    elif np.isclose(finite.max(), finite.min()):
        normalized = np.where(np.isfinite(matrix), 0.5, np.nan)
    else:
        normalized = (matrix - finite.min()) / (finite.max() - finite.min())

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#efefef")
    im = ax.imshow(np.ma.masked_invalid(normalized), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#444444")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            norm_val = normalized[i, j]
            txt_color = "white" if np.isfinite(norm_val) and norm_val > 0.58 else "black"
            stroke_color = "black" if txt_color == "white" else "white"
            ax.text(
                j,
                i,
                f"{val:.02f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=7.2,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.1, foreground=stroke_color)],
            )

    if best_mode is not None:
        highlight_extreme_cells(ax, matrix, row_keys, mode=best_mode, exclude=exclude_best)
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)


def load_suite(results_dir: Path) -> dict[str, dict]:
    suite = {}
    for stem, _ in DGP_SPECS:
        path = results_dir / f"{stem}.json"
        if path.exists():
            data = load_json(path)
            summary = extract_summary(data)
            merged = {"summary": summary, "raw_data": data}

            if stem in SYNTH_EXTRA_FILES:
                for extra_name in SYNTH_EXTRA_FILES[stem]:
                    extra_path = results_dir / extra_name
                    if not extra_path.exists():
                        continue
                    extra_data = load_json(extra_path)
                    extra_summary = extract_summary(extra_data)
                    merged["summary"] = {**merged["summary"], **extra_summary}
                    merged.setdefault("extra_raw_data", {})[extra_name] = extra_data

            suite[stem] = merged
    return suite


def load_real_suite(results_dir: Path) -> dict[str, dict]:
    suite = {}
    for filename, task in REAL_SPECS:
        path = results_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        summary = extract_summary(data)
        merged = {"summary": summary, "raw_data": data}

        if task in REAL_EXTRA_FILES:
            for extra_name in REAL_EXTRA_FILES[task]:
                extra_path = results_dir / extra_name
                if not extra_path.exists():
                    continue
                extra_data = load_json(extra_path)
                extra_summary = extract_summary(extra_data)
                merged["summary"] = {**merged["summary"], **extra_summary}
                merged.setdefault("extra_raw_data", {})[extra_name] = extra_data

        suite[task] = merged
    return suite


def fig1_allocation_geometry(suite: dict[str, dict], output_dir: Path):
    """Illustrate simplex allocation failure on the smooth-scale synthetic regime."""
    stem = "d2_pure_scale"
    task_path = REPO_ROOT / "release" / "simplextasks-12" / "synthetic" / stem / "task.npz"
    if stem not in suite or not task_path.exists():
        print("Skipping Fig 1 allocation geometry: D2 task or summary missing")
        return

    task = np.load(task_path)
    U = task["U"]
    sigma_true = task["sigma_true"]
    xy = simplex_to_xy(U)
    summary = suite[stem]["summary"]
    strata_keys = sorted(summary["global"]["stratified_coverage"].keys(), key=int)
    x = np.arange(len(strata_keys))

    rng = np.random.default_rng(2026)
    sample_idx = rng.choice(len(U), size=min(2500, len(U)), replace=False)
    xy_sample = xy[sample_idx]
    sigma_sample = sigma_true[sample_idx]

    fig = plt.figure(figsize=(8.0, 2.95), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.92, 1.18])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    sc = ax0.scatter(
        xy_sample[:, 0],
        xy_sample[:, 1],
        c=sigma_sample,
        cmap="viridis",
        s=8,
        alpha=0.8,
        linewidths=0.0,
    )
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax0.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=1.0)
    ax0.text(-0.03, -0.03, r"$u_1$", ha="right", va="top")
    ax0.text(1.03, -0.03, r"$u_2$", ha="left", va="top")
    ax0.text(0.5, np.sqrt(3.0) / 2.0 + 0.04, r"$u_3$", ha="center", va="bottom")
    ax0.set_title("D2 local scale on the simplex")
    ax0.set_aspect("equal")
    ax0.set_xlim(-0.08, 1.08)
    ax0.set_ylim(-0.08, np.sqrt(3.0) / 2.0 + 0.1)
    ax0.axis("off")
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label(r"True local scale $\sigma(u)$")

    target = 0.9
    global_cov = [summary["global"]["stratified_coverage"][k]["mean"] for k in strata_keys]
    ax1.bar(x, global_cov, color=METHOD_COLORS["global"], alpha=0.88, width=0.72)
    ax1.axhline(target, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Global CP allocates poorly")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{k}" for k in strata_keys])
    ax1.set_xlabel(r"Boundary strata ($S0 \rightarrow S4$)")
    ax1.set_ylabel("Coverage by stratum")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    comparison_methods = ["global", "partition", "twostage"]
    for method in comparison_methods:
        vals = [summary[method]["stratified_coverage"][k]["mean"] for k in strata_keys]
        ax2.plot(
            x,
            vals,
            color=METHOD_COLORS[method],
            linewidth=1.8,
            marker=PROFILE_MARKERS.get(method, "o"),
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
    ax2.axhline(target, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Repair depends on the regime")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{k}" for k in strata_keys])
    ax2.set_xlabel(r"Boundary strata ($S0 \rightarrow S4$)")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax2.legend(loc="lower right", frameon=False)

    for ax, label in zip([ax0, ax1, ax2], ["A", "B", "C"]):
        ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=11, fontweight="bold")

    save_figure(fig, output_dir, "fig1_allocation_geometry.pdf")
    plt.close(fig)


def fig1_disparity_heatmap(suite: dict[str, dict], output_dir: Path):
    """Heatmap of max disparity across synthetic regimes and methods."""
    methods = [method for method in METHOD_ORDER if method != "weighted"]
    disparity = np.full((len(methods), len(DGP_SPECS)), np.nan)

    for j, (stem, _) in enumerate(DGP_SPECS):
        if stem not in suite:
            continue
        summary = extract_summary(suite[stem])
        for i, method in enumerate(methods):
            if method in summary:
                disparity[i, j] = metric_mean(summary, method, "max_disparity")

    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    col_labels = [label for _, label in DGP_SPECS]
    row_labels = [METHOD_LABELS[m] for m in methods]
    draw_metric_heatmap(
        fig,
        ax,
        disparity,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Max disparity",
        best_mode=None,
        row_keys=methods,
        exclude_best={"oracle"},
    )

    save_figure(fig, output_dir, "fig1_synthetic_disparity_heatmap.pdf")
    plt.close(fig)


def _plot_strata_panel(ax, data: dict, methods: list[str], title: str):
    raw = data.get("raw_data", data)
    config = raw.get("config", {})
    evaluation = config.get("evaluation", {})
    alpha = evaluation.get("alpha", config.get("alpha", 0.1))
    target = 1.0 - float(alpha)
    summary = extract_summary(raw)

    reference_method = next(iter(summary))
    strata_keys = sorted(summary[reference_method]["stratified_coverage"].keys(), key=int)
    x = np.arange(len(strata_keys))
    for method in methods:
        if method not in summary:
            continue
        vals = [summary[method]["stratified_coverage"][k]["mean"] for k in strata_keys]
        ax.plot(
            x,
            vals,
            color=METHOD_COLORS[method],
            linewidth=1.6,
            marker=PROFILE_MARKERS.get(method, "o"),
            markersize=4.2,
            label=METHOD_LABELS[method],
        )

    ax.axhline(target, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in strata_keys])
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.set_ylabel("Coverage by stratum")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)


def fig2_stratified_profiles(suite: dict[str, dict], output_dir: Path):
    """Representative stratified coverage plots for key regimes."""
    panels = [
        ("d2_pure_scale", ["global", "twostage", "oracle"], "D2: Smooth Scale"),
        ("d3_discrete_groups_aligned", ["global", "partition", "oracle"], "D3: Aligned Discrete"),
        ("d6_high_k", ["global", "partition", "twostage", "oracle"], "D6: High-K"),
    ]
    available_panels = [(stem, methods, title) for stem, methods, title in panels if stem in suite]
    if not available_panels:
        print("Skipping Fig 2: no matching synthetic results found")
        return

    fig, axes = plt.subplots(1, len(available_panels), figsize=(3.4 * len(available_panels), 2.9), sharey=True, constrained_layout=True)
    if len(available_panels) == 1:
        axes = [axes]

    for ax, (stem, methods, title) in zip(axes, available_panels):
        _plot_strata_panel(ax, suite[stem], methods, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)

    save_figure(fig, output_dir, "fig2_stratified_profiles.pdf")
    plt.close(fig)


def fig3_regime_scatter(suite: dict[str, dict], output_dir: Path):
    """Scatter of worst-stratum coverage vs disparity for selected methods."""
    selected_methods = ["global", "partition", "twostage", "fullcp", "jackknife_plus", "oracle"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for stem, label in DGP_SPECS:
        if stem not in suite:
            continue
        summary = extract_summary(suite[stem])
        for method in selected_methods:
            if method not in summary:
                continue
            x = metric_mean(summary, method, "max_disparity")
            y = metric_mean(summary, method, "worst_stratum_coverage")
            ax.scatter(
                x,
                y,
                s=65,
                color=METHOD_COLORS[method],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(x + 0.01, y, label.replace("\n", " "), fontsize=7, alpha=0.9)

    ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Max disparity")
    ax.set_ylabel("Worst-stratum coverage")
    ax.set_title("Synthetic Regimes: Fairness-Safety Tradeoff")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=METHOD_LABELS[m],
                   markerfacecolor=METHOD_COLORS[m], markeredgecolor="white", markersize=8)
        for m in selected_methods
        if any(stem in suite and m in suite[stem]["summary"] for stem, _ in DGP_SPECS)
    ]
    ax.legend(handles=legend_handles, ncol=3, loc="lower left")

    save_figure(fig, output_dir, "fig3_regime_tradeoff.pdf")
    plt.close(fig)


def fig4_runtime_tradeoff(suite: dict[str, dict], output_dir: Path):
    """Runtime versus disparity on the smooth-scale benchmark."""
    stem = "d2_pure_scale"
    if stem not in suite:
        print("Skipping Fig 4: d2_pure_scale.json not found")
        return

    summary = extract_summary(suite[stem])
    methods = available_methods(summary)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    for method in methods:
        x = metric_mean(summary, method, "runtime_sec")
        y = metric_mean(summary, method, "max_disparity")
        ax.scatter(
            x,
            y,
            s=55,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            marker=REAL_MARKERS.get(method, "o"),
        )
        ax.text(x * 1.05 if x > 0 else x + 0.02, y, METHOD_LABELS[method], fontsize=7.3, va="center")

    ax.set_xlabel("Mean runtime per repetition (sec)")
    ax.set_ylabel("Max disparity")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.7)

    save_figure(fig, output_dir, "fig4_runtime_tradeoff.pdf")
    plt.close(fig)


def fig5_real_disparity_heatmap(real_suite: dict[str, dict], output_dir: Path):
    methods = ["global", "partition", "twostage", "fullcp", "jackknife_plus", "oneshot", "trainres"]
    tasks = [task for _, task in REAL_SPECS if task in real_suite]
    if not tasks:
        print("Skipping Fig 5: no real-data results found")
        return

    matrix = np.full((len(methods), len(tasks)), np.nan)
    for j, task in enumerate(tasks):
        summary = real_suite[task]["summary"]
        for i, method in enumerate(methods):
            if method in summary and "max_disparity" in summary[method]:
                matrix[i, j] = metric_mean(summary, method, "max_disparity")

    task_labels = {
        "CIFAR-10": "CIFAR-10",
        "Samson": "Samson",
        "Topics": "Topics",
        "AffectiveText": "Affective\nText",
        "UTKFace": "UTKFace",
        "PBMC": "PBMC",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    col_labels = [task_labels.get(task, task) for task in tasks]
    row_labels = [METHOD_LABELS[m] for m in methods]
    draw_metric_heatmap(
        fig,
        ax,
        matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Max disparity",
        best_mode=None,
        row_keys=methods,
    )

    save_figure(fig, output_dir, "fig5_real_disparity_heatmap.pdf")
    plt.close(fig)


def fig6_real_tradeoff(real_suite: dict[str, dict], output_dir: Path):
    selected_methods = ["global", "partition", "twostage", "jackknife_plus", "fullcp", "trainres"]
    tasks = [task for _, task in REAL_SPECS if task in real_suite]
    if not tasks:
        print("Skipping Fig 6: no real-data results found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), sharey=True)
    axes = axes.flatten()
    used_methods = set()

    for idx, ax in enumerate(axes):
        if idx >= len(tasks):
            ax.axis("off")
            continue
        task = tasks[idx]
        summary = real_suite[task]["summary"]
        xs = []
        for method in selected_methods:
            if method not in summary or "mean_radius" not in summary[method]:
                continue
            x = metric_mean(summary, method, "mean_radius")
            y = metric_mean(summary, method, "max_disparity")
            xerr = metric_std(summary, method, "mean_radius")
            yerr = metric_std(summary, method, "max_disparity")
            xs.append(x)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=METHOD_COLORS[method],
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.28,
                zorder=1,
            )
            ax.scatter(
                x,
                y,
                s=42,
                marker=REAL_MARKERS.get(method, "o"),
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.96,
                zorder=2,
            )
            used_methods.add(method)

        if xs:
            xmin = min(xs)
            xmax = max(xs)
            span = xmax - xmin
            pad = 0.08 * span if span > 0 else max(0.05, 0.15 * xmax)
            ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
        ax.set_ylim(0.0, 0.95)
        ax.set_title(task)
        ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.7)
        if idx % 3 == 0:
            ax.set_ylabel("Max disparity")
        if idx >= 3:
            ax.set_xlabel("Mean radius")

    handles = [
        plt.Line2D(
            [0], [0],
            marker=REAL_MARKERS.get(method, "o"),
            color=METHOD_COLORS[method],
            linestyle="None",
            label=METHOD_LABELS[method],
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markersize=6.5,
        )
        for method in selected_methods
        if method in used_methods
    ]
    fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.28, wspace=0.16)
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)

    save_figure(fig, output_dir, "fig6_real_tradeoff.pdf")
    plt.close(fig)


def fig_metric_sensitivity_heatmaps(real_suite: dict[str, dict], output_dir: Path):
    """Appendix diagnostic: real-task heatmaps under alternative summary metrics."""
    methods = ["global", "partition", "twostage", "fullcp", "jackknife_plus", "oneshot", "trainres"]
    tasks = [task for _, task in REAL_SPECS if task in real_suite]
    if not tasks:
        print("Skipping metric sensitivity heatmaps: no real-data results found")
        return

    panels = [
        ("max_disparity", "Max disparity"),
        ("worst_stratum_coverage", "Worst-stratum coverage"),
        ("mean_radius", "Mean radius"),
        ("sscv", "SSCV"),
    ]
    task_labels = {
        "CIFAR-10": "CIFAR-10",
        "Samson": "Samson",
        "Topics": "Topics",
        "AffectiveText": "Affective\nText",
        "UTKFace": "UTKFace",
        "PBMC": "PBMC",
    }

    fig, axes_grid = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    axes = axes_grid.ravel()
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#efefef")

    for ax, (metric, title) in zip(axes, panels):
        matrix = np.full((len(methods), len(tasks)), np.nan)
        for j, task in enumerate(tasks):
            summary = real_suite[task]["summary"]
            for i, method in enumerate(methods):
                if method in summary and metric in summary[method]:
                    matrix[i, j] = metric_mean(summary, method, metric)

        finite = matrix[np.isfinite(matrix)]
        if len(finite) == 0:
            normalized = matrix.copy()
        elif np.isclose(finite.max(), finite.min()):
            normalized = np.where(np.isfinite(matrix), 0.5, np.nan)
        else:
            normalized = (matrix - finite.min()) / (finite.max() - finite.min())

        ax.imshow(np.ma.masked_invalid(normalized), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([task_labels.get(task, task) for task in tasks], rotation=35, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_xticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(methods)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                norm_val = normalized[i, j]
                txt_color = "white" if np.isfinite(norm_val) and norm_val > 0.58 else "black"
                stroke_color = "black" if txt_color == "white" else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.02f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=6.5,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.0, foreground=stroke_color)],
                )

    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.88, pad=0.02)
    cbar.set_ticks([])

    save_figure(fig, output_dir, "fig_metric_sensitivity_heatmaps.pdf")
    plt.close(fig)


def fig_synthetic_metric_sensitivity_heatmaps(suite: dict[str, dict], output_dir: Path):
    """Appendix diagnostic: synthetic-task heatmaps under alternative summary metrics."""
    methods = ["global", "partition", "twostage", "fullcp", "jackknife_plus", "oneshot", "trainres"]
    tasks = [stem for stem, _ in DGP_SPECS if stem in suite]
    if not tasks:
        print("Skipping synthetic metric sensitivity heatmaps: no synthetic results found")
        return

    panels = [
        ("max_disparity", "Max disparity"),
        ("worst_stratum_coverage", "Worst-stratum coverage"),
        ("mean_radius", "Mean radius"),
        ("sscv", "SSCV"),
    ]
    task_labels = {stem: label.replace("\n", " ") for stem, label in DGP_SPECS}

    fig, axes_grid = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    axes = axes_grid.ravel()
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#efefef")

    for ax, (metric, title) in zip(axes, panels):
        matrix = np.full((len(methods), len(tasks)), np.nan)
        for j, task in enumerate(tasks):
            summary = suite[task]["summary"]
            for i, method in enumerate(methods):
                if method in summary and metric in summary[method]:
                    matrix[i, j] = metric_mean(summary, method, metric)

        finite = matrix[np.isfinite(matrix)]
        if len(finite) == 0:
            normalized = matrix.copy()
        elif np.isclose(finite.max(), finite.min()):
            normalized = np.where(np.isfinite(matrix), 0.5, np.nan)
        else:
            normalized = (matrix - finite.min()) / (finite.max() - finite.min())

        ax.imshow(np.ma.masked_invalid(normalized), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([task_labels.get(task, task) for task in tasks], rotation=35, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_xticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(methods)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                norm_val = normalized[i, j]
                txt_color = "white" if np.isfinite(norm_val) and norm_val > 0.58 else "black"
                stroke_color = "black" if txt_color == "white" else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.02f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=6.5,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.0, foreground=stroke_color)],
                )

    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.88, pad=0.02)
    cbar.set_ticks([])

    save_figure(fig, output_dir, "fig_synthetic_metric_sensitivity_heatmaps.pdf")
    plt.close(fig)


def fig7_real_profiles(real_suite: dict[str, dict], output_dir: Path):
    panels = [
        ("CIFAR-10", ["global", "partition", "jackknife_plus"], "CIFAR-10"),
        ("Topics", ["global", "twostage", "jackknife_plus"], "Topics"),
        ("AffectiveText", ["global", "partition", "fullcp"], "AffectiveText"),
        ("UTKFace", ["global", "partition", "jackknife_plus"], "UTKFace"),
        ("PBMC", ["global", "partition", "twostage"], "PBMC"),
    ]
    available = [(task, methods, title) for task, methods, title in panels if task in real_suite]
    if not available:
        print("Skipping Fig 7: no matching real-data results found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=True)
    axes = axes.flatten()

    for idx, (task, methods, title) in enumerate(available):
        ax = axes[idx]
        summary = real_suite[task]["summary"]
        alpha = 0.1
        target = 1.0 - alpha
        reference_method = next(iter(summary))
        strata_keys = sorted(summary[reference_method]["stratified_coverage"].keys(), key=int)
        x = np.arange(len(strata_keys))

        for method in methods:
            if method not in summary:
                continue
            vals = [summary[method]["stratified_coverage"][k]["mean"] for k in strata_keys]
            ax.plot(
                x,
                vals,
                color=METHOD_COLORS[method],
                linewidth=1.5,
                marker=PROFILE_MARKERS.get(method, "o"),
                markersize=4.0,
                label=METHOD_LABELS[method],
            )

        ax.axhline(target, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{k}" for k in strata_keys])
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
        if idx % 3 == 0:
            ax.set_ylabel("Coverage by stratum")

    for idx in range(len(available), len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.25, wspace=0.15)
    fig.legend(handles, labels, ncol=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)

    save_figure(fig, output_dir, "fig7_real_profiles.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/tables")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument(
        "--fig",
        default="all",
        help="Which figure: lead,1,2,3,4,5,6,7,metrics,synthetic-metrics,synthetic,real,all",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite(results_dir)
    real_suite = load_real_suite(results_dir)

    if args.fig in ("lead", "all", "synthetic"):
        fig1_allocation_geometry(suite, output_dir)
    if args.fig in ("1", "all", "synthetic"):
        fig1_disparity_heatmap(suite, output_dir)
    if args.fig in ("2", "all", "synthetic"):
        fig2_stratified_profiles(suite, output_dir)
    if args.fig in ("3", "all", "synthetic"):
        fig3_regime_scatter(suite, output_dir)
    if args.fig in ("4", "all", "synthetic"):
        fig4_runtime_tradeoff(suite, output_dir)
    if args.fig in ("synthetic-metrics", "all", "synthetic"):
        fig_synthetic_metric_sensitivity_heatmaps(suite, output_dir)
    if args.fig in ("5", "all", "real"):
        fig5_real_disparity_heatmap(real_suite, output_dir)
    if args.fig in ("6", "all", "real"):
        fig6_real_tradeoff(real_suite, output_dir)
    if args.fig in ("metrics", "all", "real"):
        fig_metric_sensitivity_heatmaps(real_suite, output_dir)
    if args.fig in ("7", "all", "real"):
        fig7_real_profiles(real_suite, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
