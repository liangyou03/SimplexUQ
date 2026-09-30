"""Generate appendix-ready LaTeX tables from benchmark summaries."""
from __future__ import annotations

import json
from pathlib import Path

from make_tables import EXTRA_FILES, METHOD_LABELS, METHOD_ORDER, REAL_SPECS, extract_summary, load_json

SUMMARY_METHODS = ["global", "partition", "twostage", "fullcp", "jackknife_plus", "oneshot", "trainres", "weighted"]
PBMC_RUNTIME_FALLBACK = "pbmc_sensitivity_exp2_1_bulk_deconv_boundary_fixed.json"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def merge_summary(label: str, filename: str, results_dir: Path) -> dict:
    summary = extract_summary(load_json(results_dir / filename))
    if label in EXTRA_FILES:
        for extra_name in EXTRA_FILES[label]:
            extra_path = results_dir / extra_name
            if extra_path.exists():
                summary = {**summary, **extract_summary(load_json(extra_path))}
    return summary


def format_cov(entry: dict | None) -> str:
    if not entry:
        return "--"
    return f"{entry['mean']:.3f}"


def metric_mean(entry: dict | None, key: str) -> str:
    if not entry:
        return "--"
    value = entry.get(key)
    if value is None:
        return "--"
    if isinstance(value, dict):
        return f"{value['mean']:.3f}"
    return f"{float(value):.3f}"


def worst_cov(entry: dict | None) -> str:
    if not entry:
        return "--"
    if "worst_stratum_coverage" in entry:
        return metric_mean(entry, "worst_stratum_coverage")
    stratified = entry.get("stratified_coverage", {})
    if not stratified:
        return "--"
    vals = []
    for value in stratified.values():
        if isinstance(value, dict):
            vals.append(float(value["mean"]))
        else:
            vals.append(float(value))
    return "--" if not vals else f"{min(vals):.3f}"


def runtime_mean(summary: dict, method: str, label: str, results_dir: Path) -> str:
    entry = summary.get(method)
    if entry and "runtime_sec" in entry:
        return metric_mean(entry, "runtime_sec")
    if label == "PBMC":
        fallback_path = results_dir / PBMC_RUNTIME_FALLBACK
        if fallback_path.exists():
            fallback_summary = extract_summary(load_json(fallback_path))
            fallback_entry = fallback_summary.get(method)
            if fallback_entry and "runtime_sec" in fallback_entry:
                return metric_mean(fallback_entry, "runtime_sec")
    return "--"


def build_real_summary_tables(results_dir: Path) -> list[str]:
    blocks: list[str] = []
    task_groups = [REAL_SPECS[:3], REAL_SPECS[3:]]
    for table_idx, specs in enumerate(task_groups, start=1):
        blocks.append("\\begin{table*}[!htbp]")
        blocks.append("\\centering")
        blocks.append(
            "\\caption{Real-task summary metrics. Rows report mean marginal coverage, worst-stratum coverage, max disparity, mean radius, and runtime per repetition for each default real-data benchmark task and method. PBMC runtime entries use the corresponding boundary-based rerun because the original deconvolution summary files predated runtime logging for several split methods.}"
            if table_idx == 1 else
            "\\caption[]{Real-task summary metrics (continued).}"
        )
        blocks.append(f"\\label{{tab:real-summary-{table_idx}}}")
        blocks.append("\\scriptsize")
        blocks.append("\\setlength{\\tabcolsep}{4pt}")
        blocks.append("\\begin{tabular}{@{}llccccc@{}}")
        blocks.append("\\toprule")
        blocks.append("Task & Method & Coverage & Worst & Disparity & Radius & Runtime (s) \\\\")
        blocks.append("\\midrule")

        first_task = True
        for filename, label in specs:
            path = results_dir / filename
            if not path.exists():
                continue
            summary = merge_summary(label, filename, results_dir)
            methods_present = [m for m in SUMMARY_METHODS if m in summary]
            if not methods_present:
                continue
            if not first_task:
                blocks.append("\\midrule")
            first_task = False
            for row_idx, method in enumerate(methods_present):
                entry = summary.get(method)
                task_cell = (
                    f"\\multirow{{{len(methods_present)}}}{{*}}{{{latex_escape(label)}}}"
                    if row_idx == 0 else ""
                )
                row = [
                    task_cell,
                    METHOD_LABELS[method],
                    metric_mean(entry, "marginal_coverage"),
                    worst_cov(entry),
                    metric_mean(entry, "max_disparity"),
                    metric_mean(entry, "mean_radius"),
                    runtime_mean(summary, method, label, results_dir),
                ]
                blocks.append(" & ".join(row) + " \\\\")
        blocks.append("\\bottomrule")
        blocks.append("\\end{tabular}")
        blocks.append("\\end{table*}")
        blocks.append("")
    return blocks


def build_per_strata_tables(results_dir: Path) -> list[str]:
    blocks: list[str] = []
    for filename, label in REAL_SPECS:
        path = results_dir / filename
        if not path.exists():
            continue
        summary = merge_summary(label, filename, results_dir)
        ref_method = next((m for m in SUMMARY_METHODS if m in summary), None)
        if ref_method is None:
            continue
        strata_keys = sorted(summary[ref_method]["stratified_coverage"].keys(), key=int)
        cols = "l" + "c" * len(SUMMARY_METHODS)
        blocks.append("\\begin{table*}[!htbp]")
        blocks.append("\\centering")
        blocks.append(
            f"\\caption{{Per-stratum coverage for {latex_escape(label)}. Entries report mean empirical coverage across repetitions.}}"
        )
        blocks.append(f"\\label{{tab:strata-{label.lower().replace(' ', '-')}}}")
        blocks.append("\\small")
        blocks.append(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
        blocks.append("\\toprule")
        headers = ["Stratum"] + [METHOD_LABELS[m] for m in SUMMARY_METHODS]
        blocks.append(" & ".join(headers) + " \\\\")
        blocks.append("\\midrule")
        for k in strata_keys:
            row = [f"S{k}"]
            for method in SUMMARY_METHODS:
                entry = summary.get(method, {}).get("stratified_coverage", {}).get(k)
                row.append(format_cov(entry))
            blocks.append(" & ".join(row) + " \\\\")
        blocks.append("\\bottomrule")
        blocks.append("\\end{tabular}")
        blocks.append("\\end{table*}")
        blocks.append("")
    return blocks


def build_runtime_table(results_dir: Path) -> list[str]:
    blocks: list[str] = []
    blocks.append("\\begin{table*}[!htbp]")
    blocks.append("\\centering")
    blocks.append("\\caption{Mean runtime per repetition in seconds on the real-data benchmark. Entries marked \\texttt{--} were not run for that task.}")
    blocks.append("\\label{tab:runtime-real}")
    blocks.append("\\small")
    cols = "l" + "c" * len(REAL_SPECS)
    blocks.append(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    blocks.append("\\toprule")
    blocks.append("Method & " + " & ".join(label for _, label in REAL_SPECS) + " \\\\")
    blocks.append("\\midrule")
    for method in SUMMARY_METHODS:
        row = [METHOD_LABELS[method]]
        for filename, label in REAL_SPECS:
            path = results_dir / filename
            if not path.exists():
                row.append("--")
                continue
            summary = merge_summary(label, filename, results_dir)
            runtime = runtime_mean(summary, method, label, results_dir)
            row.append(runtime if runtime == "--" else f"{float(runtime):.2f}")
        blocks.append(" & ".join(row) + " \\\\")
    blocks.append("\\bottomrule")
    blocks.append("\\end{tabular}")
    blocks.append("\\end{table*}")
    blocks.append("")
    return blocks


def main() -> None:
    results_dir = Path("results/tables")
    out_path = Path("paper/rewrite_2026/latex/generated_appendix_tables.tex")
    blocks: list[str] = []
    blocks.append("% Auto-generated by scripts/make_appendix_tables.py")
    blocks.append("\\FloatBarrier")
    blocks.append("\\subsection{Real-task summary tables}")
    blocks.append("")
    blocks.extend(build_real_summary_tables(results_dir))
    blocks.append("\\clearpage")
    blocks.append("\\FloatBarrier")
    blocks.append("\\subsection{Per-strata coverage tables}")
    blocks.append("")
    blocks.extend(build_per_strata_tables(results_dir))
    blocks.append("\\clearpage")
    blocks.append("\\FloatBarrier")
    blocks.append("\\subsection{Runtime comparison}")
    blocks.append("")
    blocks.extend(build_runtime_table(results_dir))
    out_path.write_text("\n".join(blocks) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
