"""Exp 2.5 — Topic proportion prediction on 20 Newsgroups.

Train LDA to get topic proportions (ground truth), then predict from TF-IDF features.
Output ∈ Δ^{K-1} where K = number of topics.

No external data download needed — sklearn has 20 Newsgroups built in.

Usage:
    python scripts/run_topics.py --K 10
    python scripts/run_topics.py --K 20
"""
import argparse
import json
import logging
import numpy as np
from pathlib import Path
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


def prepare_topic_data(K: int = 10, n_features: int = 5000, seed: int = 2026):
    """Build topic proportion prediction task from 20 Newsgroups.

    Pipeline:
    1. Load 20 Newsgroups, compute TF-IDF
    2. Fit LDA with K topics -> get document-topic proportions (ground truth Y)
    3. Train a regression model from TF-IDF -> topic proportions (predictions U)

    Returns:
        Y: ground truth topic proportions (n, K)
        U: predicted topic proportions (n, K)
        X_tfidf: TF-IDF features (n, n_features)
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)

    # Load data
    log.info("Loading 20 Newsgroups...")
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = newsgroups.data
    log.info(f"  {len(texts)} documents")

    # Count vectorizer for LDA
    log.info("Fitting count vectorizer...")
    count_vec = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features, stop_words="english")
    X_counts = count_vec.fit_transform(texts)

    # TF-IDF for prediction features
    log.info("Computing TF-IDF features...")
    tfidf_vec = TfidfVectorizer(max_df=0.95, min_df=2,
                                 max_features=n_features, stop_words="english")
    X_tfidf = tfidf_vec.fit_transform(texts)

    # Fit LDA -> ground truth proportions
    log.info(f"Fitting LDA with K={K} topics...")
    lda = LatentDirichletAllocation(
        n_components=K, random_state=seed, max_iter=20,
        learning_method="online", batch_size=256,
    )
    Y = lda.fit_transform(X_counts)  # (n, K), rows sum to 1
    # Ensure simplex
    Y = np.maximum(Y, 1e-8)
    Y = Y / Y.sum(axis=1, keepdims=True)

    log.info(f"  Topic proportions: shape={Y.shape}, "
             f"entropy range=[{(-Y * np.log(Y)).sum(1).min():.2f}, "
             f"{(-Y * np.log(Y)).sum(1).max():.2f}]")

    # Predict topic proportions from TF-IDF using kNN
    log.info("Training kNN predictor for topic proportions...")
    n = len(texts)
    train_idx = rng.choice(n, size=int(0.6 * n), replace=False)
    test_mask = np.ones(n, dtype=bool)
    test_mask[train_idx] = False

    X_dense = X_tfidf.toarray()

    # Use PCA to reduce dimensionality for kNN
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=seed)
    X_reduced = svd.fit_transform(X_tfidf)

    knn = KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=-1)
    knn.fit(X_reduced[train_idx], Y[train_idx])
    U = knn.predict(X_reduced)
    U = np.maximum(U, 1e-8)
    U = U / U.sum(axis=1, keepdims=True)

    # Only use test portion for evaluation
    Y_eval = Y[test_mask]
    U_eval = U[test_mask]

    log.info(f"  Evaluation set: {len(Y_eval)} documents")

    return Y_eval, U_eval


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
        raise ValueError("Non-fixed topic strata must be 'boundary' or 'entropy'.")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10, help="Number of LDA topics")
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

    # Prepare data
    Y, U = prepare_topic_data(K=args.K, seed=args.seed)
    K = Y.shape[1]

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
    log.info(f"RESULTS — Topic proportions (K={K})")
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
    out_file = out_dir / f"exp2_5_topics_K{K}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(dict(summary=summary, K=K, n=len(Y),
                       config=vars(args), raw=all_results), f, indent=2)
    log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
