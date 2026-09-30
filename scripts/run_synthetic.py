"""Run a synthetic SimplexUQ benchmark experiment from config."""
import argparse
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
from src.methods import (
    full_conformal,
    global_split_conformal,
    jackknife_plus_conformal,
    oneshot_conformal,
    oracle_conformal,
    partition_conformal,
    trainres_conformal,
    twostage_conformal,
    weighted_conformal,
)
from src.methods._knn_sigma import knn_sigma_hat, knn_sigma_leave_one_out
from src.metrics import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    mean_radius,
    mean_volume_ratio,
    size_stratified_coverage_violation,
    stratified_coverage,
    volume_ratio_by_strata,
    worst_stratum_coverage,
)
from src.utils.seed import get_rng
from src.utils.strata import (
    precompute_fixed_strata,
    stratify_by_argmax_group,
    stratify_by_boundary,
    stratify_by_entropy,
    stratify_by_kmeans,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DGP_MAP = {
    "pure_scale": PureScaleDGP,
    "model_bias": ModelBiasDGP,
    "discrete_groups": DiscreteGroupsDGP,
    "heavy_tail": HeavyTailDGP,
    "high_k": HighKDGP,
}

STRATA_MAP = {
    "boundary": stratify_by_boundary,
    "entropy": stratify_by_entropy,
    "kmeans": stratify_by_kmeans,
}

DEFAULT_METHODS = [
    "global",
    "fullcp",
    "jackknife_plus",
    "partition",
    "twostage",
    "oneshot",
    "trainres",
    "weighted",
    "oracle",
]


def get_strata(U: np.ndarray, cfg: dict) -> np.ndarray:
    method = cfg["evaluation"]["strata_method"]
    if method == "argmax_group":
        split_index = cfg["evaluation"].get("split_index", cfg["dgp"].get("easy_classes", 5))
        return stratify_by_argmax_group(U, split_index=split_index)
    return STRATA_MAP[method](U, cfg["evaluation"]["n_strata"])


def get_fixed_strata_labels(U: np.ndarray, cfg: dict) -> np.ndarray:
    """Compute one prediction-space stratification map before cal/test splitting."""
    method = cfg["evaluation"]["strata_method"]
    if method == "argmax_group":
        split_index = cfg["evaluation"].get("split_index", cfg["dgp"].get("easy_classes", 5))
        return stratify_by_argmax_group(U, split_index=split_index)
    return precompute_fixed_strata(
        U,
        method,
        cfg["evaluation"]["n_strata"],
        seed=cfg.get("seed", 2026),
    )


def get_alpha(cfg: dict) -> float:
    if "conformal" in cfg:
        return cfg["conformal"]["alpha"]
    return cfg["evaluation"]["alpha"]


def get_methods(cfg: dict) -> list[str]:
    if "conformal" in cfg and "methods" in cfg["conformal"]:
        return list(cfg["conformal"]["methods"])
    return list(cfg.get("methods", DEFAULT_METHODS))


def get_method_params(cfg: dict, method_name: str) -> dict:
    return dict(cfg.get("method_params", {}).get(method_name, {}))


def compute_weight_vectors(
    cfg: dict,
    R_cal: np.ndarray,
    U_cal: np.ndarray,
    U_test: np.ndarray,
    sigma_cal_true: np.ndarray | None,
    sigma_test_true: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build weighted-CP importance weights.

    Defaults to inverse local scale so that hard regions receive more mass.
    """
    weight_cfg = cfg.get("weighting", {})
    mode = weight_cfg.get("mode", "inverse_sigma")
    source = weight_cfg.get("source", "knn")
    eps = float(weight_cfg.get("eps", 1e-8))

    if mode != "inverse_sigma":
        raise ValueError(f"Unsupported weighting mode: {mode}")

    if source == "oracle" and sigma_cal_true is not None and sigma_test_true is not None:
        sigma_cal = sigma_cal_true
        sigma_test = sigma_test_true
    elif source == "knn_loo":
        sigma_cal = knn_sigma_leave_one_out(U_cal, R_cal, k=weight_cfg.get("k", 20))
        sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=weight_cfg.get("k", 20))
    else:
        sigma_cal = knn_sigma_hat(U_cal, R_cal, U_cal, k=weight_cfg.get("k", 20))
        sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=weight_cfg.get("k", 20))

    weights_cal = 1.0 / np.maximum(sigma_cal, eps)
    weights_test = 1.0 / np.maximum(sigma_test, eps)

    # Weighted conformal depends only on relative weights; normalize for stability.
    weights_cal = weights_cal / np.mean(weights_cal)
    weights_test = weights_test / np.mean(weights_test)
    return weights_cal, weights_test


def evaluate_method(
    res,
    U_test: np.ndarray,
    strata_test: np.ndarray,
    alpha: float,
    runtime_sec: float,
    cfg: dict,
) -> dict:
    metrics = {
        "marginal_coverage": float(marginal_coverage(res.covered)),
        "max_disparity": float(max_disparity(res.covered, strata_test, alpha)),
        "worst_stratum_coverage": float(worst_stratum_coverage(res.covered, strata_test)),
        "mean_radius": float(mean_radius(res.radius)),
        "sscv": float(size_stratified_coverage_violation(res.covered, res.radius, alpha)),
        "coverage_variance": float(coverage_variance(res.covered, strata_test)),
        "runtime_sec": float(runtime_sec),
        "stratified_coverage": {
            str(k): float(v) for k, v in stratified_coverage(res.covered, strata_test).items()
        },
    }
    volume_cfg = cfg["evaluation"].get("volume", {})
    if volume_cfg.get("compute", False):
        score = volume_cfg.get("score", "aitchison")
        n_mc = int(volume_cfg.get("n_mc", 20000))
        max_points = volume_cfg.get("max_points")
        seed = int(volume_cfg.get("seed", 0))
        metrics["mean_volume_ratio"] = float(
            mean_volume_ratio(
                U_test,
                res.radius,
                score=score,
                n_mc=n_mc,
                max_points=max_points,
                rng=np.random.default_rng(seed),
            )
        )
        metrics["volume_ratio_by_strata"] = {
            str(k): float(v)
            for k, v in volume_ratio_by_strata(
                U_test,
                res.radius,
                strata_test,
                score=score,
                n_mc=n_mc,
                max_points=max_points,
                rng=np.random.default_rng(seed),
            ).items()
        }
    return metrics


def run_one_rep(dgp, cfg: dict, rng: np.random.Generator) -> dict:
    """Generate one synthetic draw and evaluate the requested methods."""
    dcfg = cfg["data"]
    ecfg = cfg["evaluation"]
    alpha = get_alpha(cfg)

    n_total = dcfg["n_cal"] + dcfg["n_test"]
    sample = dgp.sample(n_total, rng)

    idx = rng.permutation(n_total)
    idx_cal = idx[:dcfg["n_cal"]]
    idx_test = idx[dcfg["n_cal"]:]

    R_cal, R_test = sample.R[idx_cal], sample.R[idx_test]
    U_cal, U_test = sample.U[idx_cal], sample.U[idx_test]
    sigma_cal = sample.sigma_true[idx_cal] if sample.sigma_true is not None else None
    sigma_test = sample.sigma_true[idx_test] if sample.sigma_true is not None else None

    fixed_strata = get_fixed_strata_labels(sample.U, cfg)
    strata_cal = fixed_strata[idx_cal]
    strata_test = fixed_strata[idx_test]
    weights_cal, weights_test = compute_weight_vectors(
        cfg, R_cal, U_cal, U_test, sigma_cal, sigma_test
    )

    results = {}
    methods = get_methods(cfg)

    for method_name in methods:
        method_params = get_method_params(cfg, method_name)
        start = time.perf_counter()

        if method_name == "global":
            res = global_split_conformal(R_cal, R_test, alpha)
        elif method_name == "partition":
            res = partition_conformal(R_cal, R_test, alpha, strata_cal, strata_test)
        elif method_name == "twostage":
            res = twostage_conformal(
                R_cal,
                R_test,
                alpha,
                U_cal,
                U_test,
                n_scale_est=dcfg.get("n_scale_est"),
                k=method_params.get("k", 20),
            )
        elif method_name == "fullcp":
            res = full_conformal(
                R_cal,
                R_test,
                alpha,
                U_cal,
                U_test,
                k=method_params.get("k", 20),
            )
        elif method_name == "jackknife_plus":
            res = jackknife_plus_conformal(
                R_cal,
                R_test,
                alpha,
                U_cal=U_cal,
                U_test=U_test,
                loo_scores=None,
                k=method_params.get("k", 20),
            )
        elif method_name == "oracle":
            if sigma_cal is None or sigma_test is None:
                log.warning("Skipping oracle: true sigma unavailable")
                continue
            res = oracle_conformal(R_cal, R_test, alpha, sigma_cal, sigma_test)
        elif method_name == "oneshot":
            res = oneshot_conformal(
                R_cal,
                R_test,
                alpha,
                U_cal,
                U_test,
                k=method_params.get("k", 20),
            )
        elif method_name == "trainres":
            n_train = dcfg.get("n_train", dcfg["n_cal"])
            train_sample = dgp.sample(n_train, rng)
            res = trainres_conformal(
                R_cal,
                R_test,
                alpha,
                U_cal,
                U_test,
                train_sample.R,
                train_sample.U,
                k=method_params.get("k", 20),
            )
        elif method_name == "weighted":
            res = weighted_conformal(R_cal, R_test, alpha, weights_cal, weights_test)
        else:
            log.warning("Unknown method: %s", method_name)
            continue

        runtime_sec = time.perf_counter() - start
        results[method_name] = evaluate_method(
            res,
            U_test,
            strata_test,
            alpha,
            runtime_sec=runtime_sec,
            cfg=cfg,
        )

    return results


def aggregate_repetitions(all_results: dict[str, list[dict]]) -> dict:
    """Aggregate scalar and per-stratum metrics over repetitions."""
    scalar_keys = [
        "marginal_coverage",
        "max_disparity",
        "worst_stratum_coverage",
        "mean_radius",
        "sscv",
        "coverage_variance",
        "runtime_sec",
        "mean_volume_ratio",
    ]

    summary = {}
    for method_name, reps in all_results.items():
        if not reps:
            continue

        summary[method_name] = {}
        for key in scalar_keys:
            if key in reps[0]:
                values = [rep[key] for rep in reps]
                summary[method_name][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        all_strata_keys = set()
        for rep in reps:
            all_strata_keys.update(rep["stratified_coverage"].keys())

        summary[method_name]["stratified_coverage"] = {}
        for stratum in sorted(all_strata_keys, key=int):
            values = [
                rep["stratified_coverage"][stratum]
                for rep in reps
                if stratum in rep["stratified_coverage"]
            ]
            summary[method_name]["stratified_coverage"][stratum] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_reps": len(values),
            }

        if "volume_ratio_by_strata" in reps[0]:
            all_vol_keys = set()
            for rep in reps:
                all_vol_keys.update(rep["volume_ratio_by_strata"].keys())
            summary[method_name]["volume_ratio_by_strata"] = {}
            for stratum in sorted(all_vol_keys, key=int):
                values = [
                    rep["volume_ratio_by_strata"][stratum]
                    for rep in reps
                    if stratum in rep["volume_ratio_by_strata"]
                ]
                summary[method_name]["volume_ratio_by_strata"][stratum] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_reps": len(values),
                }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("Experiment: %s", cfg["experiment"])

    dgp_cfg = cfg["dgp"]
    dgp_name = dgp_cfg["name"]
    dgp_cls = DGP_MAP[dgp_name]
    dgp_kwargs = {k: v for k, v in dgp_cfg.items() if k != "name"}
    dgp = dgp_cls(**dgp_kwargs)

    n_rep = cfg["data"]["n_rep"]
    methods = get_methods(cfg)
    rng = get_rng(cfg["seed"])
    all_results = {method: [] for method in methods}

    for rep_idx in range(n_rep):
        rep_results = run_one_rep(dgp, cfg, rng)
        for method_name, metrics in rep_results.items():
            all_results[method_name].append(metrics)

        if (rep_idx + 1) % 25 == 0 or rep_idx == n_rep - 1:
            log.info("  Completed %d/%d reps", rep_idx + 1, n_rep)

    summary = aggregate_repetitions(all_results)

    log.info("")
    log.info("=" * 72)
    log.info("RESULTS (mean +- std over %d reps)", n_rep)
    log.info("=" * 72)
    for method_name in methods:
        if method_name not in summary:
            continue
        cov = summary[method_name]["marginal_coverage"]["mean"]
        disp = summary[method_name]["max_disparity"]["mean"]
        worst = summary[method_name]["worst_stratum_coverage"]["mean"]
        radius = summary[method_name]["mean_radius"]["mean"]
        sscv = summary[method_name]["sscv"]["mean"]
        log.info(
            "  %-14s cov=%.3f disp=%.3f worst=%.3f radius=%.3f sscv=%.3f",
            method_name,
            cov,
            disp,
            worst,
            radius,
            sscv,
        )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['experiment']}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "config": cfg,
                "methods": methods,
                "summary": summary,
                "raw": all_results,
            },
            f,
            indent=2,
        )
    log.info("Saved to %s", out_file)


if __name__ == "__main__":
    main()
