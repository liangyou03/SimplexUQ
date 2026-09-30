"""Exp 2.4 — Label distribution learning: facial age estimation.

Predict age distribution (soft label over age bins) from facial images.
Ground truth is a Gaussian-smoothed label over K age bins.

Uses UTKFace dataset (free, no registration needed).

Usage:
    python scripts/run_age_ldl.py --data-dir data/raw/UTKFace
    python scripts/run_age_ldl.py --data-dir data/raw/UTKFace --K 10
"""
import argparse
import json
import logging
import numpy as np
from pathlib import Path
import re
import time

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
]


def age_to_soft_label(age: int, K: int = 10, age_range: tuple = (0, 100),
                      sigma: float = 2.0) -> np.ndarray:
    """Convert integer age to Gaussian-smoothed distribution over K bins.

    Args:
        age: integer age
        K: number of bins
        age_range: (min_age, max_age)
        sigma: smoothing in bin units

    Returns:
        distribution over K bins, sums to 1
    """
    bin_edges = np.linspace(age_range[0], age_range[1], K + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    # Gaussian kernel centered at true age
    probs = np.exp(-0.5 * ((bin_centers - age) / (sigma * bin_width)) ** 2)
    probs = probs / probs.sum()
    # Floor for numerical safety
    probs = np.maximum(probs, 1e-8)
    probs = probs / probs.sum()
    return probs


def load_utkface(data_dir: str, K: int = 10, sigma: float = 2.0):
    """Load UTKFace dataset and create soft labels.

    UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg

    Download from: https://susanqq.github.io/UTKFace/
    Or: kaggle datasets download jangedoo/utkface-new

    Returns:
        ages: integer ages (n,)
        Y: soft labels (n, K)
        image_paths: list of paths (for optional feature extraction)
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))

    if not files:
        raise FileNotFoundError(
            f"No images found in {data_dir}. "
            "Download UTKFace from https://susanqq.github.io/UTKFace/"
        )

    ages = []
    valid_files = []
    for f in files:
        # Parse age from filename
        parts = f.stem.split("_")
        try:
            age = int(parts[0])
            if 0 <= age <= 100:
                ages.append(age)
                valid_files.append(f)
        except (ValueError, IndexError):
            continue

    ages = np.array(ages)
    Y = np.array([age_to_soft_label(a, K=K, sigma=sigma) for a in ages])

    log.info(f"Loaded {len(ages)} images, age range [{ages.min()}, {ages.max()}]")
    log.info(f"Soft labels: K={K}, sigma={sigma}")

    return ages, Y, valid_files


def extract_image_features(
    image_paths: list,
    image_size: int = 16,
    cache_name: str | None = None,
):
    """Extract compact image features from UTKFace files.

    The representation is intentionally lightweight: RGB thumbnail pixels plus
    a few global summary statistics. This keeps the benchmark CPU-friendly
    while making the predictor depend on image content rather than the target
    age metadata.
    """
    from PIL import Image

    cache_path = None
    if cache_name is not None:
        cache_path = Path("data/processed") / cache_name
        if cache_path.exists():
            log.info(f"Loading cached UTKFace image features from {cache_path}")
            return np.load(cache_path)["X"]

    feats = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        rgb_flat = arr.reshape(-1)
        gray = arr.mean(axis=2)
        stats = np.array([
            gray.mean(),
            gray.std(),
            arr[..., 0].mean(),
            arr[..., 1].mean(),
            arr[..., 2].mean(),
        ], dtype=np.float32)
        feats.append(np.concatenate([rgb_flat, stats]))

    X = np.asarray(feats, dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X)
        log.info(f"Cached UTKFace image features to {cache_path}")
    return X


def get_age_predictions(ages: np.ndarray, Y: np.ndarray, image_paths: list,
                        K: int, method: str = "knn", seed: int = 2026):
    """Get predicted age distributions.

    Methods:
    - 'knn': use age as feature, kNN regression in label space (diagnostic baseline)
    - 'image_knn': use thumbnail image features + PCA + kNN regression
    - 'noisy': add noise to true labels (controlled experiment)
    - 'cnn': train a CNN (requires GPU, optional)

    Returns:
        U: predicted distributions (n, K)
    """
    if method == "noisy":
        # Add heteroscedastic noise: more noise for middle ages
        rng = np.random.default_rng(seed)
        noise_scale = 0.05 + 0.15 * np.abs(ages - 50) / 50  # more noise at extremes
        noise = rng.normal(0, noise_scale[:, None], Y.shape)
        U = Y + noise
        U = np.maximum(U, 1e-8)
        U = U / U.sum(axis=1, keepdims=True)
        return U

    elif method == "image_knn":
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cache_name = f"utkface_imgfeat_{len(image_paths)}_s16.npz"
        X = extract_image_features(image_paths, image_size=16, cache_name=cache_name)

        rng = np.random.default_rng(seed)
        n = len(ages)
        train_idx = rng.choice(n, size=int(0.8 * n), replace=False)

        pca_dim = min(64, X.shape[1], len(train_idx))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=pca_dim, random_state=seed),
            KNeighborsRegressor(n_neighbors=25, weights="distance"),
        )
        model.fit(X[train_idx], Y[train_idx])
        U = model.predict(X)
        U = np.maximum(U, 1e-8)
        U = U / U.sum(axis=1, keepdims=True)
        return U

    elif method == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        # Use age as the sole feature, predict soft label
        X = ages.reshape(-1, 1)
        # Leave-one-out style: train on 80%, predict on all
        rng = np.random.default_rng(seed)
        n = len(ages)
        train_idx = rng.choice(n, size=int(0.8 * n), replace=False)
        model = KNeighborsRegressor(n_neighbors=20, weights="distance")
        model.fit(X[train_idx], Y[train_idx])
        U = model.predict(X)
        U = np.maximum(U, 1e-8)
        U = U / U.sum(axis=1, keepdims=True)
        return U

    elif method == "cnn":
        raise ValueError(
            "CNN predictor training is outside this fixed-predictor artifact. "
            "Use 'image_knn', 'knn', or 'noisy'."
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_weight_vectors(R_cal, U_cal, U_test, k=20):
    sigma_cal = knn_sigma_leave_one_out(U_cal, R_cal, k=k)
    sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=k)
    weights_cal = 1.0 / np.maximum(sigma_cal, 1e-8)
    weights_test = 1.0 / np.maximum(sigma_test, 1e-8)
    weights_cal /= np.mean(weights_cal)
    weights_test /= np.mean(weights_test)
    return weights_cal, weights_test


def run_experiment(
    Y,
    U,
    alpha,
    n_rep,
    cal_frac,
    n_strata,
    rng,
    methods,
    compute_volume=False,
    volume_score="aitchison",
    volume_n_mc=50000,
    volume_max_points=None,
    strata_method="entropy",
    fixed_strata=True,
    strata_seed=2026,
):
    """Standard conformal experiment."""
    R = aitchison_dist(Y, U)
    n = len(R)
    n_cal = int(n * cal_frac)

    all_results = {m: [] for m in methods}
    fixed_labels = None
    if fixed_strata:
        fixed_labels = precompute_fixed_strata(U, strata_method, n_strata, seed=strata_seed)
    elif strata_method not in {"boundary", "entropy"}:
        raise ValueError("Non-fixed age strata must be 'boundary' or 'entropy'.")

    for rep in range(n_rep):
        perm = rng.permutation(n)
        idx_cal, idx_test = perm[:n_cal], perm[n_cal:]

        R_cal, R_test = R[idx_cal], R[idx_test]
        U_cal, U_test = U[idx_cal], U[idx_test]

        if fixed_labels is not None:
            strata_cal = fixed_labels[idx_cal]
            strata_test = fixed_labels[idx_test]
        else:
            strata_fn = stratify_by_boundary if strata_method == "boundary" else stratify_by_entropy
            strata_cal = strata_fn(U_cal, n_strata)
            strata_test = strata_fn(U_test, n_strata)
        weights_cal, weights_test = compute_weight_vectors(R_cal, U_cal, U_test)

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


def maybe_subsample(ages, Y, image_paths, max_samples, rng):
    if max_samples is None or max_samples >= len(Y):
        return ages, Y, image_paths
    idx = rng.choice(len(Y), size=max_samples, replace=False)
    idx = np.sort(idx)
    return ages[idx], Y[idx], [image_paths[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/UTKFace")
    parser.add_argument("--K", type=int, default=10, help="Number of age bins")
    parser.add_argument("--sigma", type=float, default=2.0, help="Label smoothing width")
    parser.add_argument(
        "--pred-method",
        default="image_knn",
        choices=["image_knn", "knn", "noisy", "cnn"],
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_rep", type=int, default=200)
    parser.add_argument("--cal_frac", type=float, default=0.4)
    parser.add_argument("--n_strata", type=int, default=5)
    parser.add_argument(
        "--strata",
        choices=["entropy", "boundary", "dominant", "kmeans", "random"],
        default="entropy",
    )
    parser.add_argument("--fixed-strata", dest="fixed_strata", action="store_true")
    parser.add_argument(
        "--separate-strata",
        dest="fixed_strata",
        action="store_false",
        help="Diagnostic only: fit calibration/test strata separately.",
    )
    parser.set_defaults(fixed_strata=True)
    parser.add_argument("--max_samples", type=int, default=None)
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
    parser.add_argument("--volume-n-mc", type=int, default=50000)
    parser.add_argument("--volume-max-points", type=int, default=None)
    args = parser.parse_args()

    rng = get_rng(args.seed)

    # Load data
    ages, Y, image_paths = load_utkface(args.data_dir, K=args.K, sigma=args.sigma)
    ages, Y, image_paths = maybe_subsample(ages, Y, image_paths, args.max_samples, rng)

    # Get predictions
    log.info(f"Getting predictions (method={args.pred_method})...")
    U = get_age_predictions(ages, Y, image_paths, K=args.K,
                            method=args.pred_method, seed=args.seed)

    R = aitchison_dist(Y, U)
    log.info(f"Residuals: mean={R.mean():.4f}, std={R.std():.4f}")

    # Run
    all_results = run_experiment(
        Y,
        U,
        args.alpha,
        args.n_rep,
        args.cal_frac,
        args.n_strata,
        rng,
        args.methods,
        compute_volume=args.compute_volume,
        volume_score=args.volume_score,
        volume_n_mc=args.volume_n_mc,
        volume_max_points=args.volume_max_points,
        strata_method=args.strata,
        fixed_strata=args.fixed_strata,
        strata_seed=args.seed,
    )

    # Report
    log.info("\n" + "=" * 60)
    log.info(f"RESULTS — Age LDL (K={args.K}, method={args.pred_method})")
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
            f"disp={s['max_disparity']['mean']:.3f}±{s['max_disparity']['std']:.3f}"
        )

    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"exp2_4_age_ldl_K{args.K}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(dict(summary=summary, K=args.K, n=len(ages),
                       config=vars(args), raw=all_results), f, indent=2)
    log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
