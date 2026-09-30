"""Summarize all-real-task stratification sensitivity runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from make_tables import METHOD_LABELS


TASK_FILES = {
    "cifar10": "exp2_2_softmax_cifar10_strata_{strata}_fixed.json",
    "topics": "exp2_5_topics_K10_strata_{strata}_fixed.json",
    "affectivetext": "exp2_6_affective_text_strata_{strata}_fixed.json",
    "samson": "exp2_3_hyperspectral_samson_nmf_strata_{strata}_fixed.json",
    "utkface": "exp2_4_age_ldl_K10_strata_{strata}_fixed.json",
    "pbmc": "pbmc_sensitivity_exp2_1_bulk_deconv_{strata}_fixed.json",
}

DISPLAY_NAMES = {
    "cifar10": "CIFAR-10",
    "topics": "Topics",
    "affectivetext": "AffectiveText",
    "samson": "Samson",
    "utkface": "UTKFace",
    "pbmc": "PBMC",
}

STRATA_ORDER = ["boundary", "entropy", "dominant", "kmeans"]
EXPECTED_REPS = {
    "cifar10": 50,
    "topics": 50,
    "affectivetext": 50,
    "samson": 50,
    "utkface": 50,
    "pbmc": 200,
}


def extract_n_rep(data: dict) -> int | None:
    config = data.get("config", {})
    for key in ("n_rep", "n_reps"):
        if key in config:
            return int(config[key])
    conformal = config.get("conformal", {})
    for key in ("n_rep", "n_reps"):
        if key in conformal:
            return int(conformal[key])
    evaluation = config.get("evaluation", {})
    for key in ("n_rep", "n_reps"):
        if key in evaluation:
            return int(evaluation[key])
    return None


def load_summary(path: Path) -> tuple[dict, float, int | None]:
    data = json.loads(path.read_text())
    summary = data.get("summary", data.get("aggregated"))
    config = data.get("config", {})

    alpha = None
    if "alpha" in config:
        alpha = float(config["alpha"])
    elif "conformal" in config and "alpha" in config["conformal"]:
        alpha = float(config["conformal"]["alpha"])
    if alpha is None:
        alpha = 0.1

    return summary, alpha, extract_n_rep(data)


def metric_mean(entry: dict, key: str) -> float:
    value = entry.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", float("nan")))
    return float(value)


def rank_methods(summary: dict, alpha: float, coverage_tol: float) -> tuple[list[str], list[str]]:
    nominal = 1.0 - alpha
    methods = list(summary)

    valid = [
        method for method in methods
        if metric_mean(summary[method], "marginal_coverage") >= nominal - coverage_tol
    ]
    valid_sorted = sorted(
        valid,
        key=lambda m: (
            metric_mean(summary[m], "max_disparity"),
            metric_mean(summary[m], "mean_radius"),
            -metric_mean(summary[m], "marginal_coverage"),
        ),
    )
    raw_sorted = sorted(
        methods,
        key=lambda m: (
            metric_mean(summary[m], "max_disparity"),
            metric_mean(summary[m], "mean_radius"),
            -metric_mean(summary[m], "marginal_coverage"),
        ),
    )
    return valid_sorted, raw_sorted


def summarize(input_dir: Path, coverage_tol: float) -> tuple[list[dict], list[dict]]:
    detail_rows: list[dict] = []
    winner_rows: list[dict] = []

    for task, template in TASK_FILES.items():
        winners = []
        for strata in STRATA_ORDER:
            path = input_dir / template.format(strata=strata)
            if not path.exists():
                winner_rows.append({
                    "task": task,
                    "task_label": DISPLAY_NAMES[task],
                    "strata": strata,
                    "status": "missing",
                })
                continue

            summary, alpha, n_rep = load_summary(path)
            expected_reps = EXPECTED_REPS[task]
            if n_rep is None or n_rep < expected_reps:
                winner_rows.append({
                    "task": task,
                    "task_label": DISPLAY_NAMES[task],
                    "strata": strata,
                    "status": "incomplete",
                    "observed_n_rep": n_rep,
                    "expected_n_rep": expected_reps,
                })
                continue
            valid_rank, raw_rank = rank_methods(summary, alpha, coverage_tol)
            nominal = 1.0 - alpha

            for rank, method in enumerate(raw_rank, start=1):
                entry = summary[method]
                detail_rows.append({
                    "task": task,
                    "task_label": DISPLAY_NAMES[task],
                    "strata": strata,
                    "method": method,
                    "rank_raw": rank,
                    "is_valid": metric_mean(entry, "marginal_coverage") >= nominal - coverage_tol,
                    "marginal_coverage": metric_mean(entry, "marginal_coverage"),
                    "max_disparity": metric_mean(entry, "max_disparity"),
                    "worst_stratum_coverage": metric_mean(entry, "worst_stratum_coverage"),
                    "mean_radius": metric_mean(entry, "mean_radius"),
                    "runtime_sec": metric_mean(entry, "runtime_sec"),
                    "n_rep": n_rep,
                })

            if valid_rank:
                best = valid_rank[0]
                winners.append(best)
                best_entry = summary[best]
                winner_rows.append({
                    "task": task,
                    "task_label": DISPLAY_NAMES[task],
                    "strata": strata,
                    "status": "ok",
                    "best_valid_method": best,
                    "best_valid_coverage": metric_mean(best_entry, "marginal_coverage"),
                    "best_valid_disparity": metric_mean(best_entry, "max_disparity"),
                    "best_valid_radius": metric_mean(best_entry, "mean_radius"),
                    "best_raw_method": raw_rank[0] if raw_rank else "",
                    "valid_ranking": valid_rank,
                    "raw_ranking": raw_rank,
                })
            else:
                winner_rows.append({
                    "task": task,
                    "task_label": DISPLAY_NAMES[task],
                    "strata": strata,
                    "status": "no_valid_method",
                    "best_raw_method": raw_rank[0] if raw_rank else "",
                    "valid_ranking": [],
                    "raw_ranking": raw_rank,
                })

        if winners:
            counts = Counter(winners)
            modal, modal_count = counts.most_common(1)[0]
            winner_rows.append({
                "task": task,
                "task_label": DISPLAY_NAMES[task],
                "strata": "_summary",
                "status": "task_summary",
                "modal_best_valid_method": modal,
                "modal_count": modal_count,
                "winner_stable": modal_count == len(STRATA_ORDER),
                "winner_set": sorted(counts),
            })

    return detail_rows, winner_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, winner_rows: list[dict]) -> None:
    per_task = {}
    for row in winner_rows:
        task = row["task"]
        per_task.setdefault(task, {})
        per_task[task][row["strata"]] = row

    lines = [
        "# Real-Task Stratification Sensitivity Summary",
        "",
        "Rows report the best valid method under each fixed stratification.",
        "A method is considered valid if mean marginal coverage is at least nominal minus the configured tolerance.",
        "",
        "| Task | Boundary | Entropy | Dominant | KMeans | Stable winner? |",
        "|---|---|---|---|---|---|",
    ]

    for task in TASK_FILES:
        task_rows = per_task.get(task, {})
        cells = []
        for strata in STRATA_ORDER:
            row = task_rows.get(strata)
            if not row or row.get("status") != "ok":
                if row and row.get("status") == "incomplete":
                    cells.append("incomplete")
                else:
                    cells.append("missing")
                continue
            cells.append(
                f"{row['best_valid_method']} "
                f"({row['best_valid_disparity']:.3f}, cov={row['best_valid_coverage']:.3f})"
            )
        summary = task_rows.get("_summary", {})
        stable = summary.get("winner_stable")
        if stable is True:
            stable_text = f"yes ({summary.get('modal_best_valid_method')})"
        elif stable is False:
            stable_text = ", ".join(summary.get("winner_set", []))
        else:
            stable_text = "pending"
        lines.append(
            f"| {DISPLAY_NAMES[task]} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {stable_text} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def format_cell(row: dict | None) -> str:
    if not row or row.get("status") != "ok":
        if row and row.get("status") == "incomplete":
            return "incomplete"
        return "--"
    method = METHOD_LABELS.get(row["best_valid_method"], row["best_valid_method"])
    disparity = row["best_valid_disparity"]
    coverage = row["best_valid_coverage"]
    return f"{latex_escape(method)} ({disparity:.3f}, {coverage:.3f})"


def write_latex(path: Path, winner_rows: list[dict]) -> None:
    per_task = {}
    for row in winner_rows:
        task = row["task"]
        per_task.setdefault(task, {})
        per_task[task][row["strata"]] = row

    lines = [
        "% Auto-generated by scripts/summarize_real_strata_sensitivity.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Real-task stratification sensitivity across fixed alternative strata. Each cell reports the best valid method under that stratification, shown as method name with $(\\text{max disparity}, \\text{marginal coverage})$. A method is treated as valid when mean marginal coverage is at least nominal minus the configured tolerance.}",
        "\\label{tab:real-strata-sensitivity}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccp{3.3cm}@{}}",
        "\\toprule",
        "Task & Boundary & Entropy & Dominant & KMeans & Winner stability \\\\",
        "\\midrule",
    ]

    for task in TASK_FILES:
        task_rows = per_task.get(task, {})
        summary = task_rows.get("_summary", {})
        stable = summary.get("winner_stable")
        if stable is True:
            stability_text = f"Stable: {METHOD_LABELS.get(summary.get('modal_best_valid_method', ''), summary.get('modal_best_valid_method', ''))}"
        elif stable is False:
            winners = [
                METHOD_LABELS.get(name, name)
                for name in summary.get("winner_set", [])
            ]
            stability_text = "Mixed: " + ", ".join(winners)
        else:
            stability_text = "Pending"

        row = [
            latex_escape(DISPLAY_NAMES[task]),
            format_cell(task_rows.get("boundary")),
            format_cell(task_rows.get("entropy")),
            format_cell(task_rows.get("dominant")),
            format_cell(task_rows.get("kmeans")),
            latex_escape(stability_text),
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/tables")
    parser.add_argument("--coverage-tol", type=float, default=0.02)
    parser.add_argument(
        "--detail-csv",
        default="results/tables/real_strata_sensitivity_detail.csv",
    )
    parser.add_argument(
        "--winner-json",
        default="results/tables/real_strata_sensitivity_winners.json",
    )
    parser.add_argument(
        "--winner-md",
        default="results/tables/real_strata_sensitivity_summary.md",
    )
    parser.add_argument(
        "--winner-tex",
        default="paper/rewrite_2026/latex/generated_real_strata_sensitivity.tex",
    )
    args = parser.parse_args()

    detail_rows, winner_rows = summarize(Path(args.input_dir), args.coverage_tol)

    write_csv(Path(args.detail_csv), detail_rows)
    Path(args.winner_json).write_text(json.dumps(winner_rows, indent=2), encoding="utf-8")
    write_markdown(Path(args.winner_md), winner_rows)
    write_latex(Path(args.winner_tex), winner_rows)


if __name__ == "__main__":
    main()
