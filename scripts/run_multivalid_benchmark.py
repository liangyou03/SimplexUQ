"""Run paired Global, Mondrian, and BatchMVP on SimplexTasks-12."""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dgp.discrete_groups import DiscreteGroupsDGP
from src.dgp.heavy_tail import HeavyTailDGP
from src.dgp.high_k import HighKDGP
from src.dgp.model_bias import ModelBiasDGP
from src.dgp.pure_scale import PureScaleDGP
from src.methods import batch_mvp_conformal, global_split_conformal, partition_conformal
from src.metrics import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    mean_radius,
    size_stratified_coverage_violation,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.utils.multivalid_groups import build_multivalid_groups
from src.utils.simplex import aitchison_dist
from src.utils.strata import precompute_fixed_strata, stratify_by_argmax_group


log = logging.getLogger(__name__)

DGP_MAP = {
    "pure_scale": PureScaleDGP,
    "model_bias": ModelBiasDGP,
    "discrete_groups": DiscreteGroupsDGP,
    "heavy_tail": HeavyTailDGP,
    "high_k": HighKDGP,
}

SCALAR_METRICS = (
    "marginal_coverage",
    "max_disparity",
    "worst_stratum_coverage",
    "mean_radius",
    "sscv",
    "coverage_variance",
    "overlap_max_disparity",
    "overlap_worst_coverage",
    "runtime_sec",
    "n_patches",
    "n_groups",
    "max_calibration_violation",
)


def load_benchmark_config(path: Path) -> dict:
    """Load and minimally validate the benchmark registry."""
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not cfg.get("tasks"):
        raise ValueError("benchmark config must contain tasks")
    return cfg


def paired_split(
    n: int, cal_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return one disjoint random split shared by every method."""
    if not 0.0 < cal_frac < 1.0:
        raise ValueError("cal_frac must lie in (0, 1)")
    permutation = rng.permutation(n)
    n_cal = int(round(cal_frac * n))
    return permutation[:n_cal], permutation[n_cal:]


def compute_real_scores(Y: np.ndarray, U: np.ndarray, score: str) -> np.ndarray:
    """Compute the task-declared residual score from frozen arrays."""
    if Y.shape != U.shape:
        raise ValueError("Y and U must have matching shapes")
    if score.lower() in {"tv", "total_variation"}:
        return 0.5 * np.sum(np.abs(Y - U), axis=1)
    if score.lower() == "aitchison":
        return aitchison_dist(Y, U)
    raise ValueError(f"unsupported real-task score: {score}")


def _fixed_synthetic_strata(U: np.ndarray, cfg: dict) -> np.ndarray:
    """Compute the paper's fixed response-free synthetic evaluation map."""
    method = cfg["evaluation"]["strata_method"]
    if method == "argmax_group":
        split_index = cfg["evaluation"].get(
            "split_index", cfg["dgp"].get("easy_classes", 5)
        )
        return stratify_by_argmax_group(U, split_index)
    return precompute_fixed_strata(
        U,
        method,
        cfg["evaluation"]["n_strata"],
        seed=cfg.get("seed", 2026),
    )


def _overlap_metrics(
    covered: np.ndarray,
    membership: np.ndarray,
    names: tuple[str, ...],
    alpha: float,
) -> tuple[float, float, dict[str, float]]:
    """Measure coverage allocation over an overlapping response-free family."""
    target = 1.0 - alpha
    coverages = {
        name: float(covered[membership[:, index]].mean())
        for index, name in enumerate(names)
        if np.any(membership[:, index])
    }
    return (
        max(abs(value - target) for value in coverages.values()),
        min(coverages.values()),
        coverages,
    )


def _evaluate(
    result,
    labels_test: np.ndarray,
    groups_test: np.ndarray,
    group_names: tuple[str, ...],
    alpha: float,
    runtime_sec: float,
) -> dict:
    """Evaluate one method on default strata and the overlapping group family."""
    overlap_disparity, overlap_worst, overlap_coverages = _overlap_metrics(
        result.covered, groups_test, group_names, alpha
    )
    return {
        "marginal_coverage": float(marginal_coverage(result.covered)),
        "max_disparity": float(max_disparity(result.covered, labels_test, alpha)),
        "worst_stratum_coverage": float(worst_stratum_coverage(result.covered, labels_test)),
        "mean_radius": float(mean_radius(result.radius)),
        "sscv": float(
            size_stratified_coverage_violation(result.covered, result.radius, alpha)
        ),
        "coverage_variance": float(coverage_variance(result.covered, labels_test)),
        "overlap_max_disparity": float(overlap_disparity),
        "overlap_worst_coverage": float(overlap_worst),
        "runtime_sec": float(runtime_sec),
        "n_patches": int(getattr(result, "n_patches", 0)),
        "n_groups": int(getattr(result, "n_groups", groups_test.shape[1])),
        "max_calibration_violation": float(
            getattr(result, "max_violation", 0.0)
        ),
        "stratified_coverage": {
            str(key): float(value)
            for key, value in stratified_coverage(result.covered, labels_test).items()
        },
        "overlap_coverage": overlap_coverages,
    }


def _run_methods(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    labels_cal: np.ndarray,
    labels_test: np.ndarray,
    groups_cal: np.ndarray,
    groups_test: np.ndarray,
    group_names: tuple[str, ...],
    cfg: dict,
) -> dict[str, dict]:
    """Run the three paired methods on a shared split."""
    alpha = float(cfg["alpha"])
    results = {}
    for method in cfg["methods"]:
        start = time.perf_counter()
        if method == "Global":
            result = global_split_conformal(R_cal, R_test, alpha)
        elif method == "Mondrian":
            result = partition_conformal(
                R_cal, R_test, alpha, labels_cal, labels_test
            )
        elif method == "BatchMVP":
            options = cfg["multivalid"]
            result = batch_mvp_conformal(
                R_cal,
                R_test,
                alpha,
                groups_cal,
                groups_test,
                num_grid=int(options["num_grid"]),
                max_rounds=int(options["max_rounds"]),
            )
        else:
            raise ValueError(f"unknown method: {method}")
        results[method] = _evaluate(
            result,
            labels_test,
            groups_test,
            group_names,
            alpha,
            time.perf_counter() - start,
        )
    return results


def _build_groups(U: np.ndarray, labels: np.ndarray, cfg: dict):
    """Build the configured group family once on a complete response-free draw."""
    options = cfg["multivalid"]
    return build_multivalid_groups(
        U,
        labels,
        n_bins=int(options["n_bins"]),
        dominant_groups=int(options["dominant_groups"]),
        min_group_fraction=float(options["min_group_fraction"]),
        seed=int(cfg["seed"]),
    )


def _run_synthetic_task(task: dict, cfg: dict, n_rep: int) -> list[dict]:
    """Regenerate and evaluate one synthetic task for every repetition."""
    with Path(task["config"]).open(encoding="utf-8") as handle:
        source_cfg = yaml.safe_load(handle)
    dgp_cfg = dict(source_cfg["dgp"])
    dgp = DGP_MAP[dgp_cfg.pop("name")](**dgp_cfg)
    n_cal = int(source_cfg["data"]["n_cal"])
    n_test = int(source_cfg["data"]["n_test"])
    rng = np.random.default_rng(int(cfg["seed"]))
    repetitions = []
    for rep in range(n_rep):
        sample = dgp.sample(n_cal + n_test, rng)
        labels = _fixed_synthetic_strata(sample.U, source_cfg)
        groups = _build_groups(sample.U, labels, cfg)
        idx_cal, idx_test = paired_split(
            len(sample.R), n_cal / (n_cal + n_test), rng
        )
        repetitions.append(
            _run_methods(
                sample.R[idx_cal],
                sample.R[idx_test],
                labels[idx_cal],
                labels[idx_test],
                groups.membership[idx_cal],
                groups.membership[idx_test],
                groups.names,
                cfg,
            )
        )
        if (rep + 1) % 25 == 0 or rep + 1 == n_rep:
            log.info("%s: %d/%d repetitions", task["id"], rep + 1, n_rep)
    return repetitions


def _load_real_arrays(task: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only frozen Y/U arrays and the released default-strata labels."""
    with np.load(task["task"], allow_pickle=False) as archive:
        Y = np.asarray(archive["Y"], dtype=float)
        U = np.asarray(archive["U"], dtype=float)
    with Path(task["strata"]).open(encoding="utf-8") as handle:
        labels = np.asarray(json.load(handle)["labels"], dtype=int)
    if len(labels) != len(U):
        raise ValueError(f"strata length mismatch for {task['id']}")
    return Y, U, labels


def _run_real_task(task: dict, cfg: dict, n_rep: int) -> list[dict]:
    """Evaluate one frozen real task over paired random splits."""
    Y, U, labels = _load_real_arrays(task)
    scores = compute_real_scores(Y, U, task["score"])
    groups = _build_groups(U, labels, cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    repetitions = []
    for rep in range(n_rep):
        idx_cal, idx_test = paired_split(len(scores), float(task["cal_frac"]), rng)
        repetitions.append(
            _run_methods(
                scores[idx_cal],
                scores[idx_test],
                labels[idx_cal],
                labels[idx_test],
                groups.membership[idx_cal],
                groups.membership[idx_test],
                groups.names,
                cfg,
            )
        )
        if (rep + 1) % 25 == 0 or rep + 1 == n_rep:
            log.info("%s: %d/%d repetitions", task["id"], rep + 1, n_rep)
    return repetitions


def aggregate_task(repetitions: list[dict]) -> dict:
    """Aggregate scalar metrics over paired repetitions."""
    summary = {}
    for method in repetitions[0]:
        summary[method] = {}
        for metric in SCALAR_METRICS:
            values = np.asarray([rep[method][metric] for rep in repetitions], dtype=float)
            summary[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def _paired_bootstrap(
    differences: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict:
    """Return a percentile interval for a paired mean difference."""
    n = len(differences)
    indices = rng.integers(0, n, size=(n_resamples, n))
    means = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(differences)),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
    }


def paired_comparisons(repetitions: list[dict], cfg: dict) -> dict:
    """Compute paired BatchMVP-minus-baseline bootstrap intervals."""
    options = cfg["bootstrap"]
    rng = np.random.default_rng(int(cfg["seed"]) + 991)
    comparisons = {}
    for baseline in ("Global", "Mondrian"):
        comparisons[f"BatchMVP-{baseline}"] = {}
        for metric in (
            "max_disparity",
            "overlap_max_disparity",
            "worst_stratum_coverage",
            "overlap_worst_coverage",
            "mean_radius",
        ):
            differences = np.asarray(
                [rep["BatchMVP"][metric] - rep[baseline][metric] for rep in repetitions]
            )
            comparisons[f"BatchMVP-{baseline}"][metric] = _paired_bootstrap(
                differences,
                int(options["n_resamples"]),
                float(options["confidence"]),
                rng,
            )
    return comparisons


def render_summary_markdown(task_results: dict) -> str:
    """Render a compact paper/rebuttal-facing summary table."""
    lines = [
        "# SimplexTasks-12 BatchMVP summary",
        "",
        "| Task | Method | Coverage | Max disparity | Worst coverage | Mean radius | Overlap disparity | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id, result in task_results.items():
        for method, metrics in result["summary"].items():
            mean = lambda key: metrics[key]["mean"]
            lines.append(
                f"| {task_id} | {method} | {mean('marginal_coverage'):.3f} | "
                f"{mean('max_disparity'):.3f} | {mean('worst_stratum_coverage'):.3f} | "
                f"{mean('mean_radius'):.3f} | {mean('overlap_max_disparity'):.3f} | "
                f"{mean('runtime_sec'):.4f} |"
            )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, task_results: dict) -> None:
    """Write one flat summary row per task/method."""
    fields = ["task", "method"] + [
        f"{metric}_{stat}" for metric in SCALAR_METRICS for stat in ("mean", "std")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id, result in task_results.items():
            for method, metrics in result["summary"].items():
                row = {"task": task_id, "method": method}
                for metric in SCALAR_METRICS:
                    for stat in ("mean", "std"):
                        row[f"{metric}_{stat}"] = metrics[metric][stat]
                writer.writerow(row)


def _render_rebuttal(task_results: dict) -> str:
    """Render paired effect estimates for direct rebuttal use."""
    batch_valid = sum(
        result["summary"]["BatchMVP"]["marginal_coverage"]["mean"] >= 0.88
        for result in task_results.values()
    )
    wins_global_default = sum(
        result["summary"]["BatchMVP"]["max_disparity"]["mean"]
        < result["summary"]["Global"]["max_disparity"]["mean"]
        for result in task_results.values()
    )
    wins_global_overlap = sum(
        result["summary"]["BatchMVP"]["overlap_max_disparity"]["mean"]
        < result["summary"]["Global"]["overlap_max_disparity"]["mean"]
        for result in task_results.values()
    )
    wins_mondrian_overlap = sum(
        result["summary"]["BatchMVP"]["overlap_max_disparity"]["mean"]
        < result["summary"]["Mondrian"]["overlap_max_disparity"]["mean"]
        for result in task_results.values()
    )
    lines = [
        "# Paired BatchMVP comparisons",
        "",
        "## Protocol",
        "",
        "BatchMVP uses a fixed response-free overlapping group family (universal, default, entropy, boundary, and grouped dominant-component strata), a fixed monotone score transform r/(1+r), 1000 midpoint grid values, and the stopping rule from the authors' reference implementation. Every comparison uses the same calibration/test split for all methods.",
        "",
        "## Aggregate result",
        "",
        f"BatchMVP meets the paper's pre-declared mean-coverage tolerance (coverage >= 0.88 at a 0.90 target) on {batch_valid}/{len(task_results)} tasks. It reduces default-strata disparity relative to Global on {wins_global_default}/{len(task_results)} tasks and overlapping-family disparity on {wins_global_overlap}/{len(task_results)} tasks. Relative to Mondrian, it lowers overlapping-family disparity on {wins_mondrian_overlap}/{len(task_results)} tasks; Mondrian remains stronger on the single pre-declared partition overall.",
        "",
        "## Paired intervals",
        "",
        "Differences are BatchMVP minus baseline; lower disparity and higher worst coverage are better.",
        "",
        "| Task | Comparison | Metric | Mean difference | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for task_id, result in task_results.items():
        for comparison, metrics in result["paired_comparisons"].items():
            for metric, interval in metrics.items():
                lines.append(
                    f"| {task_id} | {comparison} | {metric} | {interval['mean']:.4f} | "
                    f"[{interval['low']:.4f}, {interval['high']:.4f}] |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/multivalid/simplextasks12_batchmvp.yaml",
    )
    parser.add_argument("--task", action="append", help="Run only the selected task ID")
    parser.add_argument("--n-rep", type=int, help="Override repetitions for every selected task")
    parser.add_argument(
        "--output",
        default="results/tables/multivalid_simplextasks12.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_benchmark_config(Path(args.config))
    selected = set(args.task or [])
    tasks = [task for task in cfg["tasks"] if not selected or task["id"] in selected]
    if selected - {task["id"] for task in tasks}:
        raise ValueError(f"unknown task IDs: {sorted(selected - {task['id'] for task in tasks})}")

    task_results = {}
    benchmark_start = time.perf_counter()
    for task in tasks:
        if args.n_rep is not None:
            n_rep = args.n_rep
        elif task["kind"] == "real":
            n_rep = int(task["n_rep"])
        else:
            with Path(task["config"]).open(encoding="utf-8") as handle:
                n_rep = int(yaml.safe_load(handle)["data"]["n_rep"])
        log.info("Starting %s (%d repetitions)", task["id"], n_rep)
        if task["kind"] == "synthetic":
            repetitions = _run_synthetic_task(task, cfg, n_rep)
        else:
            repetitions = _run_real_task(task, cfg, n_rep)
        task_results[task["id"]] = {
            "kind": task["kind"],
            "n_rep": n_rep,
            "summary": aggregate_task(repetitions),
            "paired_comparisons": paired_comparisons(repetitions, cfg),
            "raw": repetitions,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "elapsed_sec": time.perf_counter() - benchmark_start,
        "tasks": task_results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stem = output.with_suffix("")
    csv_path = stem.with_name(stem.name + "_summary").with_suffix(".csv")
    md_path = stem.with_name(stem.name + "_summary").with_suffix(".md")
    rebuttal_path = stem.with_name(stem.name + "_rebuttal").with_suffix(".md")
    _write_csv(csv_path, task_results)
    md_path.write_text(render_summary_markdown(task_results), encoding="utf-8")
    rebuttal_path.write_text(_render_rebuttal(task_results), encoding="utf-8")
    log.info("Saved %s and derived summaries", output)


if __name__ == "__main__":
    main()
