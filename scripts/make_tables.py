"""Generate paper-style summary tables from saved benchmark results."""
import argparse
import json
from pathlib import Path

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

METHOD_LABELS = {
    "global": "Global",
    "partition": "Mondrian",
    "twostage": "TwoStage",
    "fullcp": "FullCP",
    "jackknife_plus": "Jackknife+",
    "oneshot": "OneShot",
    "trainres": "TrainRes",
    "weighted": "Weighted",
    "oracle": "Oracle",
}

SYNTHETIC_SPECS = [
    ("d1_homogeneous.json", "D1 Homogeneous"),
    ("d2_pure_scale.json", "D2 Pure scale"),
    ("d3_discrete_groups_aligned.json", "D3 Discrete aligned"),
    ("d4_model_bias.json", "D4 Bias"),
    ("d5_heavy_tail.json", "D5 Heavy tail"),
    ("d6_high_k.json", "D6† High-K"),
]

REAL_SPECS = [
    ("exp2_2_softmax_cifar10_strata_entropy_fixed.json", "CIFAR-10"),
    ("exp2_3_hyperspectral_samson_nmf_all_methods.json", "Samson"),
    ("exp2_5_topics_K10_all_methods.json", "Topics"),
    ("exp2_6_affective_text.json", "AffectiveText"),
    ("exp2_4_age_ldl_K10_image_knn_main.json", "UTKFace"),
    ("real_bulk_deconv.json", "PBMC"),
]

EXTRA_FILES = {
    "D1 Homogeneous": ["d1_homogeneous_exact.json"],
    "D3 Discrete aligned": ["d3_discrete_groups_aux.json"],
    "D5 Heavy tail": ["d5_heavy_tail_aux.json"],
    "D6† High-K": ["d6_high_k_aux.json", "d6_high_k_exact_appendix.json"],
    "PBMC": ["real_bulk_deconv_fullcp.json", "real_bulk_deconv_aux.json", "real_bulk_deconv_trainres.json"],
    "UTKFace": ["exp2_4_age_ldl_K10_image_knn_fullcp_2k.json"],
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_summary(data: dict) -> dict:
    if "summary" in data:
        return data["summary"]
    if "aggregated" in data:
        return data["aggregated"]
    raise KeyError("Missing summary/aggregated block")


def metric_cell(summary: dict, method: str) -> str:
    if method not in summary:
        return "--"
    cov = summary[method]["marginal_coverage"]["mean"]
    disp = summary[method]["max_disparity"]["mean"]
    worst = summary[method]["worst_stratum_coverage"]["mean"] if "worst_stratum_coverage" in summary[method] else None
    if worst is None and "stratified_coverage" in summary[method]:
        worst = min(v["mean"] for v in summary[method]["stratified_coverage"].values())
    radius = summary[method]["mean_radius"]["mean"] if "mean_radius" in summary[method] else None
    if radius is None:
        return f"{cov:.3f} / {disp:.3f} / {worst:.3f}"
    return f"{cov:.3f} / {disp:.3f} / {worst:.3f} / {radius:.3f}"


def write_markdown_table(out_path: Path, title: str, specs: list[tuple[str, str]], results_dir: Path, extras: dict[str, list[str]] | None = None):
    rows = []
    for filename, label in specs:
        path = results_dir / filename
        if not path.exists():
            continue
        summary = extract_summary(load_json(path))
        if extras and label in extras:
            for extra_name in extras[label]:
                extra_path = results_dir / extra_name
                if extra_path.exists():
                    summary = {**summary, **extract_summary(load_json(extra_path))}
        row = [label]
        for method in METHOD_ORDER:
            row.append(metric_cell(summary, method))
        rows.append(row)

    headers = ["Task"] + [METHOD_LABELS[m] for m in METHOD_ORDER]
    lines = [f"# {title}", "", "Cells report `coverage / disparity / worst-stratum / radius`.", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/tables")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_markdown_table(
        output_dir / "paper_table_synthetic_summary.md",
        "Synthetic Summary",
        SYNTHETIC_SPECS,
        results_dir,
        extras=EXTRA_FILES,
    )
    write_markdown_table(
        output_dir / "paper_table_real_summary.md",
        "Real-Data Summary",
        REAL_SPECS,
        results_dir,
        extras=EXTRA_FILES,
    )


if __name__ == "__main__":
    main()
