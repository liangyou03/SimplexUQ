"""Hyperspectral unmixing experiment for conformal prediction on the simplex.

Benchmark datasets: Samson (K=3), Jasper Ridge (K=4), Urban (K=4-6)
Each pixel's abundance vector ∈ Δ^{K-1}, ground truth available.

Usage:
    python scripts/run_hyperspectral.py --dataset samson
    python scripts/run_hyperspectral.py --dataset jasper
"""
import argparse
import json
import logging
import numpy as np
from pathlib import Path
import time
from scipy.io import loadmat
from scipy.optimize import nnls

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.simplex import aitchison_dist
from src.utils.strata import (
    precompute_fixed_strata,
    stratify_by_boundary,
    stratify_by_entropy,
)
from src.utils.seed import get_rng
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
from src.metrics.coverage import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.metrics.sscv import size_stratified_coverage_violation
from src.metrics.setsize import mean_radius, mean_volume_ratio, volume_ratio_by_strata

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
    "oracle",
]

# =====================================================================
# Dataset configs
# =====================================================================
DATASETS = {
    "samson": dict(
        data_file="samson/samson_1.mat",
        endmember_file="samson/end3.mat",
        abundance_file="samson/end3.mat",  # A is in same file as M for wispcarey data
        data_key="V",          # (n_bands, n_pixels) = (156, 9025)
        endmember_key="M",     # (n_bands, K) = (156, 3)
        abundance_key="A",     # (K, n_pixels) = (3, 9025)
        n_rows=95, n_cols=95,
        K=3,
        names=["Soil", "Tree", "Water"],
    ),
    "jasper": dict(
        data_file="jasper/jasperRidge2_R198.mat",
        endmember_file="jasper/end4.mat",
        abundance_file="jasper/end4.mat",  # A is in same file as M for wispcarey data
        data_key="Y",           # (198, 10000) uint16
        endmember_key="M",      # (198, 4)
        abundance_key="A",      # (4, 10000)
        n_rows=100, n_cols=100,
        K=4,
        names=["Tree", "Water", "Dirt", "Road"],
    ),
}


def load_hyperspectral(data_dir: str, dataset: str) -> dict:
    """Load hyperspectral image, endmembers, and ground truth abundances.

    Returns:
        dict with:
            pixels: (n_pixels, n_bands) - spectral data
            endmembers: (n_bands, K) - endmember spectra
            abundances: (n_pixels, K) - ground truth fractions
            names: list of endmember names
            shape: (n_rows, n_cols)
    """
    cfg = DATASETS[dataset]
    data_dir = Path(data_dir)

    # Load image data
    img_mat = loadmat(str(data_dir / cfg["data_file"]))
    pixels = img_mat[cfg["data_key"]].astype(np.float64)
    # Ensure (n_pixels, n_bands)
    if pixels.shape[0] < pixels.shape[1]:
        pixels = pixels.T

    # Load endmembers
    end_mat = loadmat(str(data_dir / cfg["endmember_file"]))
    endmembers = end_mat[cfg["endmember_key"]].astype(np.float64)
    # Ensure (n_bands, K)
    if endmembers.shape[1] > endmembers.shape[0]:
        endmembers = endmembers.T

    # Load abundances
    abund_mat = loadmat(str(data_dir / cfg["abundance_file"]))
    abundances = abund_mat[cfg["abundance_key"]].astype(np.float64)
    # Ensure (n_pixels, K)
    if abundances.shape[0] == cfg["K"]:
        abundances = abundances.T

    n_pixels = cfg["n_rows"] * cfg["n_cols"]

    # Truncate/reshape if needed
    pixels = pixels[:n_pixels]
    abundances = abundances[:n_pixels]

    # Normalize abundances to sum to 1 (they should already, but ensure)
    row_sums = abundances.sum(axis=1, keepdims=True)
    abundances = abundances / np.maximum(row_sums, 1e-10)

    log.info(f"Dataset: {dataset}")
    log.info(f"  Pixels: {pixels.shape} ({cfg['n_rows']}x{cfg['n_cols']})")
    log.info(f"  Bands: {endmembers.shape[0]}")
    log.info(f"  Endmembers ({cfg['K']}): {cfg['names']}")
    log.info(f"  Abundance range: [{abundances.min():.4f}, {abundances.max():.4f}]")

    return dict(
        pixels=pixels,
        endmembers=endmembers,
        abundances=abundances,
        names=cfg["names"],
        shape=(cfg["n_rows"], cfg["n_cols"]),
        K=cfg["K"],
    )


def unmix_nnls(pixels: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """NNLS unmixing: for each pixel, solve min ||pixel - E @ a||^2, a >= 0.

    Args:
        pixels: (n_pixels, n_bands)
        endmembers: (n_bands, K)

    Returns:
        abundances_hat: (n_pixels, K), normalized to simplex
    """
    n = pixels.shape[0]
    K = endmembers.shape[1]
    props = np.zeros((n, K))

    for i in range(n):
        coef, _ = nnls(endmembers, pixels[i])
        total = coef.sum()
        props[i] = coef / total if total > 0 else np.ones(K) / K

    return props


def unmix_nmf(pixels: np.ndarray, K: int, seed: int = 2026) -> np.ndarray:
    """NMF-based unmixing: estimate endmembers AND abundances from data.

    Unlike NNLS with known endmembers, NMF introduces endmember estimation
    error, producing heterogeneous residuals across the simplex.

    Args:
        pixels: (n_pixels, n_bands)
        K: number of endmembers

    Returns:
        abundances_hat: (n_pixels, K), normalized to simplex
    """
    from sklearn.decomposition import NMF

    log.info(f"  Running NMF with K={K} components...")
    nmf = NMF(n_components=K, init="nndsvda", max_iter=500,
              random_state=seed, l1_ratio=0.5)
    W = nmf.fit_transform(pixels)  # (n_pixels, K) — abundance-like
    # H = nmf.components_            # (K, n_bands) — endmember-like

    # Normalize rows to simplex
    W = np.maximum(W, 1e-10)
    U = W / W.sum(axis=1, keepdims=True)

    recon_err = nmf.reconstruction_err_
    log.info(f"  NMF reconstruction error: {recon_err:.4f}")

    return U


def run_experiment(
    Y: np.ndarray,       # ground truth (n, K)
    U: np.ndarray,       # predictions (n, K)
    alpha: float,
    n_rep: int,
    cal_frac: float,
    n_strata: int,
    rng,
    methods,
    compute_volume: bool = False,
    volume_score: str = "aitchison",
    volume_n_mc: int = 20000,
    volume_max_points: int | None = None,
    strata_method: str = "boundary",
    fixed_strata: bool = True,
    strata_seed: int = 2026,
):
    """Run conformal experiment with repeated cal/test splits."""
    R = aitchison_dist(Y, U)
    n = len(R)
    n_cal = int(n * cal_frac)

    all_results = {m: [] for m in methods}
    fixed_labels = None
    if fixed_strata:
        fixed_labels = precompute_fixed_strata(U, strata_method, n_strata, seed=strata_seed)
    elif strata_method not in {"boundary", "entropy"}:
        raise ValueError("Non-fixed hyperspectral strata must be 'boundary' or 'entropy'.")

    for rep in range(n_rep):
        perm = rng.permutation(n)
        idx_cal, idx_test = perm[:n_cal], perm[n_cal:]

        R_cal, R_test = R[idx_cal], R[idx_test]
        U_cal, U_test = U[idx_cal], U[idx_test]

        if fixed_labels is not None:
            strata_cal = fixed_labels[idx_cal]
            strata_test = fixed_labels[idx_test]
        else:
            strata_fn = stratify_by_entropy if strata_method == "entropy" else stratify_by_boundary
            strata_cal = strata_fn(U_cal, n_strata)
            strata_test = strata_fn(U_test, n_strata)
        sigma_cal = knn_sigma_leave_one_out(U_cal, R_cal)
        sigma_test = knn_sigma_hat(U_cal, R_cal, U_test)
        weights_cal = 1.0 / np.maximum(sigma_cal, 1e-8)
        weights_test = 1.0 / np.maximum(sigma_test, 1e-8)
        weights_cal /= np.mean(weights_cal)
        weights_test /= np.mean(weights_test)

        for m in methods:
            start = time.perf_counter()
            if m == "global":
                res = global_split_conformal(R_cal, R_test, alpha)
            elif m == "partition":
                res = partition_conformal(R_cal, R_test, alpha,
                                          strata_cal, strata_test)
            elif m == "twostage":
                res = twostage_conformal(R_cal, R_test, alpha,
                                         U_cal, U_test)
            elif m == "jackknife_plus":
                res = jackknife_plus_conformal(R_cal, R_test, alpha, U_cal=U_cal, U_test=U_test)
            elif m == "weighted":
                res = weighted_conformal(R_cal, R_test, alpha, weights_cal, weights_test)
            elif m == "oneshot":
                res = oneshot_conformal(R_cal, R_test, alpha, U_cal, U_test)
            elif m == "trainres":
                train_perm = rng.permutation(n)
                idx_train = train_perm[:n_cal]
                res = trainres_conformal(
                    R_cal, R_test, alpha, U_cal, U_test, R[idx_train], U[idx_train]
                )
            elif m == "fullcp":
                res = full_conformal(R_cal, R_test, alpha, U_cal, U_test)
            elif m == "oracle":
                res = oracle_conformal(R_cal, R_test, alpha, sigma_cal, sigma_test)
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
                stratified_coverage={
                    str(k): float(v) for k, v in stratified_coverage(res.covered, strata_test).items()
                },
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["samson", "jasper"], default="samson")
    parser.add_argument("--data-dir", default="data/raw/hyperspectral")
    parser.add_argument("--unmix", choices=["nnls", "nmf"], default="nnls",
                        help="Unmixing method: nnls (known endmembers) or nmf (estimated)")
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
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS + ["fullcp"],
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--compute-volume", action="store_true")
    parser.add_argument("--volume-score", choices=["aitchison", "tv"], default="aitchison")
    parser.add_argument("--volume-n-mc", type=int, default=20000)
    parser.add_argument("--volume-max-points", type=int, default=None)
    args = parser.parse_args()

    # Load data
    data = load_hyperspectral(args.data_dir, args.dataset)

    # Unmix
    if args.unmix == "nmf":
        log.info("Running NMF unmixing (estimated endmembers)...")
        U = unmix_nmf(data["pixels"], data["K"], seed=args.seed)
    else:
        log.info("Running NNLS unmixing (known endmembers)...")
        U = unmix_nnls(data["pixels"], data["endmembers"])
    Y = data["abundances"]

    R = aitchison_dist(Y, U)
    log.info(f"Residuals: mean={R.mean():.4f}, std={R.std():.4f}, "
             f"median={np.median(R):.4f}")

    # Quick quality check
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(Y, U))
    corr = np.corrcoef(Y.ravel(), U.ravel())[0, 1]
    log.info(f"Unmixing quality: RMSE={rmse:.4f}, Pearson r={corr:.4f}")

    # Check heterogeneity: residuals by dominant endmember
    dominant = np.argmax(U, axis=1)
    for k in range(data["K"]):
        mask = dominant == k
        if mask.sum() > 0:
            log.info(f"  {data['names'][k]:10s}: n={mask.sum():5d}, "
                     f"R_mean={R[mask].mean():.4f}, R_std={R[mask].std():.4f}")

    # Run experiment
    rng = get_rng(args.seed)
    log.info(f"\nRunning {args.n_rep} reps, alpha={args.alpha}, "
             f"cal_frac={args.cal_frac}...")

    all_results = run_experiment(
        Y, U, args.alpha, args.n_rep, args.cal_frac,
        args.n_strata, rng, args.methods,
        compute_volume=args.compute_volume,
        volume_score=args.volume_score,
        volume_n_mc=args.volume_n_mc,
        volume_max_points=args.volume_max_points,
        strata_method=args.strata,
        fixed_strata=args.fixed_strata,
        strata_seed=args.seed,
    )

    # Aggregate
    log.info("\n" + "=" * 60)
    log.info(f"RESULTS — Hyperspectral unmixing ({args.dataset})")
    log.info("=" * 60)

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
    for m in args.methods:
        if not all_results[m]:
            continue
        reps = all_results[m]
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
        log.info(
            f"  {m:12s}  cov={s['marginal_coverage']['mean']:.3f}±{s['marginal_coverage']['std']:.3f}  "
            f"disp={s['max_disparity']['mean']:.3f}±{s['max_disparity']['std']:.3f}  "
            f"radius={s['mean_radius']['mean']:.3f}"
        )

    # Save
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.unmix}" if args.unmix != "nnls" else ""
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"exp2_3_hyperspectral_{args.dataset}{suffix}{tag_suffix}.json"
    with open(out_file, "w") as f:
        json.dump(dict(
            dataset=args.dataset,
            summary=summary,
            unmixing_rmse=float(rmse),
            unmixing_corr=float(corr),
            residual_stats=dict(mean=float(R.mean()), std=float(R.std())),
            endmember_names=data["names"],
            config=vars(args),
            raw=all_results,
        ), f, indent=2)
    log.info(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
