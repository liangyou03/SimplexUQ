"""Run conformal benchmark on SemEval-2007 Affective Text emotion compositions."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_prediction_matrix, load_affective_text
from src.methods import (
    full_conformal,
    global_split_conformal,
    jackknife_plus_conformal,
    oneshot_conformal,
    partition_conformal,
    trainres_conformal,
    twostage_conformal,
    weighted_conformal,
)
from src.methods._knn_sigma import knn_sigma_hat, knn_sigma_leave_one_out
from src.metrics.coverage import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.metrics.setsize import mean_radius, mean_volume_ratio, volume_ratio_by_strata
from src.metrics.sscv import size_stratified_coverage_violation
from src.utils.seed import get_rng
from src.utils.simplex import aitchison_dist
from src.utils.strata import (
    precompute_fixed_strata,
    stratify_by_boundary,
    stratify_by_entropy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_METHODS = [
    "global",
    "partition",
    "twostage",
    "jackknife_plus",
    "weighted",
    "oneshot",
    "trainres",
    "fullcp",
]


def compute_weight_vectors(R_cal, U_cal, U_test, k=20):
    sigma_cal = knn_sigma_leave_one_out(U_cal, R_cal, k=k)
    sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=k)
    weights_cal = 1.0 / np.maximum(sigma_cal, 1e-8)
    weights_test = 1.0 / np.maximum(sigma_test, 1e-8)
    weights_cal /= np.mean(weights_cal)
    weights_test /= np.mean(weights_test)
    return weights_cal, weights_test


def macro_pearson(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for j in range(a.shape[1]):
        aj = a[:, j]
        bj = b[:, j]
        if np.std(aj) <= 1e-12 or np.std(bj) <= 1e-12:
            continue
        vals.append(float(np.corrcoef(aj, bj)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def run_experiment(
    Y,
    U,
    alpha,
    n_rep,
    cal_frac,
    n_strata,
    rng,
    methods,
    strata_mode,
    compute_volume=False,
    volume_score="aitchison",
    volume_n_mc=50000,
    volume_max_points=None,
    fixed_strata=True,
    strata_seed=2026,
):
    R = aitchison_dist(Y, U)
    n = len(R)
    n_cal = int(n * cal_frac)
    all_results = {m: [] for m in methods}
    fixed_labels = None
    if fixed_strata:
        fixed_labels = precompute_fixed_strata(U, strata_mode, n_strata, seed=strata_seed)
    elif strata_mode not in {"boundary", "entropy"}:
        raise ValueError("Non-fixed AffectiveText strata must be 'boundary' or 'entropy'.")
    else:
        strata_fn = stratify_by_boundary if strata_mode == "boundary" else stratify_by_entropy

    for rep in range(n_rep):
        perm = rng.permutation(n)
        idx_cal, idx_test = perm[:n_cal], perm[n_cal:]

        R_cal, R_test = R[idx_cal], R[idx_test]
        U_cal, U_test = U[idx_cal], U[idx_test]

        if fixed_labels is not None:
            strata_cal = fixed_labels[idx_cal]
            strata_test = fixed_labels[idx_test]
        else:
            strata_cal = strata_fn(U_cal, n_strata)
            strata_test = strata_fn(U_test, n_strata)
        weights_cal, weights_test = compute_weight_vectors(R_cal, U_cal, U_test)

        for m in methods:
            start = time.perf_counter()
            if m == "global":
                res = global_split_conformal(R_cal, R_test, alpha)
            elif m == "partition":
                res = partition_conformal(R_cal, R_test, alpha, strata_cal, strata_test)
            elif m == "twostage":
                res = twostage_conformal(R_cal, R_test, alpha, U_cal, U_test)
            elif m == "jackknife_plus":
                res = jackknife_plus_conformal(R_cal, R_test, alpha, U_cal=U_cal, U_test=U_test)
            elif m == "weighted":
                res = weighted_conformal(R_cal, R_test, alpha, weights_cal, weights_test)
            elif m == "oneshot":
                res = oneshot_conformal(R_cal, R_test, alpha, U_cal, U_test)
            elif m == "trainres":
                train_perm = rng.permutation(n)
                idx_train = train_perm[:n_cal]
                res = trainres_conformal(R_cal, R_test, alpha, U_cal, U_test, R[idx_train], U[idx_train])
            elif m == "fullcp":
                res = full_conformal(R_cal, R_test, alpha, U_cal, U_test)
            else:
                continue
            runtime_sec = time.perf_counter() - start
            all_results[m].append(dict(
                marginal_coverage=float(marginal_coverage(res.covered)),
                max_disparity=float(max_disparity(res.covered, strata_test, alpha)),
                worst_stratum_coverage=float(worst_stratum_coverage(res.covered, strata_test)),
                mean_radius=float(mean_radius(res.radius)),
                sscv=float(size_stratified_coverage_violation(res.covered, res.radius, alpha)),
                coverage_variance=float(coverage_variance(res.covered, strata_test)),
                runtime_sec=float(runtime_sec),
                stratified_coverage={str(k): float(v) for k, v in stratified_coverage(res.covered, strata_test).items()},
            ))
            if compute_volume:
                all_results[m][-1]["mean_volume_ratio"] = float(
                    mean_volume_ratio(
                        U_test,
                        res.radius,
                        score=volume_score,
                        n_mc=volume_n_mc,
                        max_points=volume_max_points,
                        rng=np.random.default_rng(rep),
                    )
                )
                all_results[m][-1]["volume_ratio_by_strata"] = {
                    str(k): float(v)
                    for k, v in volume_ratio_by_strata(
                        U_test,
                        res.radius,
                        strata_test,
                        score=volume_score,
                        n_mc=volume_n_mc,
                        max_points=volume_max_points,
                        rng=np.random.default_rng(rep),
                    ).items()
                }
        if (rep + 1) % 50 == 0:
            log.info(f"  Rep {rep + 1}/{n_rep}")
    return all_results


def summarize_results(all_results: dict, methods: list[str]) -> dict:
    summary = {}
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
    for m in methods:
        reps = all_results.get(m, [])
        if not reps:
            continue
        s = {}
        for key in scalar_keys:
            if key in reps[0]:
                vals = [r[key] for r in reps]
                s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        strata_keys = set()
        for r in reps:
            strata_keys.update(r["stratified_coverage"].keys())
        s["stratified_coverage"] = {
            k: {
                "mean": float(np.mean([r["stratified_coverage"][k] for r in reps if k in r["stratified_coverage"]])),
                "std": float(np.std([r["stratified_coverage"][k] for r in reps if k in r["stratified_coverage"]])),
                "n_reps": int(sum(k in r["stratified_coverage"] for r in reps)),
            }
            for k in sorted(strata_keys, key=int)
        }
        if "volume_ratio_by_strata" in reps[0]:
            vol_keys = set()
            for r in reps:
                vol_keys.update(r["volume_ratio_by_strata"].keys())
            s["volume_ratio_by_strata"] = {
                k: {
                    "mean": float(np.mean([r["volume_ratio_by_strata"][k] for r in reps if k in r["volume_ratio_by_strata"]])),
                    "std": float(np.std([r["volume_ratio_by_strata"][k] for r in reps if k in r["volume_ratio_by_strata"]])),
                    "n_reps": int(sum(k in r["volume_ratio_by_strata"] for r in reps)),
                }
                for k in sorted(vol_keys, key=int)
            }
        summary[m] = s
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/AffectiveText.Semeval.2007")
    parser.add_argument("--prediction-cache", default="data/processed/affective_text_predictions.jsonl")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_rep", type=int, default=200)
    parser.add_argument("--cal_frac", type=float, default=0.4)
    parser.add_argument("--n_strata", type=int, default=5)
    parser.add_argument(
        "--strata",
        choices=["boundary", "entropy", "dominant", "kmeans", "random"],
        default="boundary",
    )
    parser.add_argument("--fixed-strata", dest="fixed_strata", action="store_true")
    parser.add_argument(
        "--separate-strata",
        dest="fixed_strata",
        action="store_false",
        help="Diagnostic only: fit calibration/test strata separately.",
    )
    parser.set_defaults(fixed_strata=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--compute-volume", action="store_true")
    parser.add_argument("--volume-score", choices=["aitchison", "tv"], default="aitchison")
    parser.add_argument("--volume-n-mc", type=int, default=50000)
    parser.add_argument("--volume-max-points", type=int, default=None)
    args = parser.parse_args()

    data = load_affective_text(args.data_dir)
    pred_raw, U = build_prediction_matrix(data["ids"], args.prediction_cache)
    Y = data["Y"]
    gold_raw = data["raw_scores"]
    rng = get_rng(args.seed)

    R = aitchison_dist(Y, U)
    macro_r = macro_pearson(gold_raw, pred_raw)
    flat_r = float(np.corrcoef(gold_raw.reshape(-1), pred_raw.reshape(-1))[0, 1])
    log.info(f"Loaded {len(Y)} headlines with cached predictions")
    log.info(f"Predictor quality: macro Pearson={macro_r:.3f}, flattened Pearson={flat_r:.3f}")
    log.info(f"Residuals: mean={R.mean():.4f}, std={R.std():.4f}")

    all_results = run_experiment(
        Y,
        U,
        args.alpha,
        args.n_rep,
        args.cal_frac,
        args.n_strata,
        rng,
        args.methods,
        args.strata,
        fixed_strata=args.fixed_strata,
        compute_volume=args.compute_volume,
        volume_score=args.volume_score,
        volume_n_mc=args.volume_n_mc,
        volume_max_points=args.volume_max_points,
        strata_seed=args.seed,
    )
    summary = summarize_results(all_results, args.methods)

    log.info("\n" + "=" * 60)
    log.info("RESULTS — SemEval-2007 Affective Text")
    log.info("=" * 60)
    for m in args.methods:
        if m not in summary:
            continue
        s = summary[m]
        log.info(
            f"  {m:12s}  cov={s['marginal_coverage']['mean']:.3f}±{s['marginal_coverage']['std']:.3f}  "
            f"disp={s['max_disparity']['mean']:.3f}±{s['max_disparity']['std']:.3f}"
        )

    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"exp2_6_affective_text{suffix}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                summary=summary,
                n=len(Y),
                K=Y.shape[1],
                emotions=data["emotions"],
                predictor_quality=dict(macro_pearson=macro_r, flattened_pearson=flat_r),
                config=vars(args),
                raw=all_results,
            ),
            f,
            indent=2,
        )
    log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
