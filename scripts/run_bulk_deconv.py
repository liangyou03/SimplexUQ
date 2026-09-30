"""Run bulk deconvolution conformal prediction experiments.

Exp 2.1: Semi-synthetic bulk RNA-seq deconvolution.
- Load PBMC3K reference
- Generate pseudo-bulk samples with known cell type proportions (ONCE)
- NNLS deconvolution → residuals (ONCE)
- 200 reps = different random cal/test splits of the same residuals
"""
import argparse
import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import scanpy as sc
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dgp.deconv import nnls_deconv
from src.dgp.pseudobulk import generate_pseudobulk
from src.methods import (
    jackknife_plus_conformal,
    global_split_conformal,
    oneshot_conformal,
    partition_conformal,
    trainres_conformal,
    weighted_conformal,
    twostage_conformal,
    full_conformal,
)
from src.methods._knn_sigma import knn_sigma_hat
from src.metrics import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.metrics import mean_radius, radius_by_strata
from src.metrics.sscv import size_stratified_coverage_violation
from src.utils.simplex import aitchison_dist
from src.utils.strata import (
    precompute_fixed_strata,
    stratify_by_boundary,
    stratify_by_entropy,
    stratify_by_kmeans,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STRATA_REGISTRY = {
    "boundary": stratify_by_boundary,
    "entropy": stratify_by_entropy,
    "kmeans": stratify_by_kmeans,
}


def run_one_rep(
    R: np.ndarray,
    U: np.ndarray,
    cfg: dict,
    rep_idx: int,
    base_seed: int,
    fixed_labels: np.ndarray | None = None,
) -> dict:
    """Run one repetition: random cal/test split + conformal methods."""
    alpha = cfg["conformal"]["alpha"]
    n_strata = cfg["evaluation"]["n_strata"]
    strata_method = cfg["evaluation"]["strata_method"]
    cal_frac = cfg["conformal"]["cal_frac"]

    n_samples = len(R)
    seed = base_seed + rep_idx
    rng = np.random.default_rng(seed)

    # Random cal/test split
    n_cal = int(n_samples * cal_frac)
    idx = rng.permutation(n_samples)
    idx_cal, idx_test = idx[:n_cal], idx[n_cal:]

    R_cal, R_test = R[idx_cal], R[idx_test]
    U_cal, U_test = U[idx_cal], U[idx_test]

    # Stratification on test set
    if fixed_labels is not None:
        strata_test = fixed_labels[idx_test]
    else:
        strata_fn = STRATA_REGISTRY[strata_method]
        strata_test = strata_fn(U_test, n_strata)

    rep_results = {}
    for method_name in cfg["conformal"]["methods"]:
        start = time.perf_counter()
        if method_name == "global":
            result = global_split_conformal(R_cal, R_test, alpha)
        elif method_name == "partition":
            if fixed_labels is not None:
                strata_cal = fixed_labels[idx_cal]
            else:
                strata_cal = strata_fn(U_cal, n_strata)
            result = partition_conformal(R_cal, R_test, alpha, strata_cal, strata_test)
        elif method_name == "twostage":
            n_scale_est = len(R_cal) // 2
            result = twostage_conformal(R_cal, R_test, alpha, U_cal, U_test, n_scale_est=n_scale_est)
        elif method_name == "fullcp":
            result = full_conformal(R_cal, R_test, alpha, U_cal, U_test)
        elif method_name == "jackknife_plus":
            result = jackknife_plus_conformal(R_cal, R_test, alpha, U_cal=U_cal, U_test=U_test)
        elif method_name == "oneshot":
            result = oneshot_conformal(R_cal, R_test, alpha, U_cal, U_test)
        elif method_name == "weighted":
            sigma_cal = knn_sigma_hat(U_cal, R_cal, U_cal)
            sigma_test = knn_sigma_hat(U_cal, R_cal, U_test)
            floor = float(np.mean(sigma_cal) * 0.1)
            weights_cal = 1.0 / np.maximum(sigma_cal, floor)
            weights_test = 1.0 / np.maximum(sigma_test, floor)
            result = weighted_conformal(R_cal, R_test, alpha, weights_cal, weights_test)
        elif method_name == "trainres":
            train_perm = rng.permutation(n_samples)
            idx_train = train_perm[:n_cal]
            result = trainres_conformal(
                R_cal, R_test, alpha, U_cal, U_test, R[idx_train], U[idx_train]
            )
        else:
            continue
        runtime_sec = time.perf_counter() - start

        rep_results[method_name] = {
            "marginal_coverage": float(marginal_coverage(result.covered)),
            "max_disparity": float(max_disparity(result.covered, strata_test, alpha)),
            "worst_stratum_coverage": float(worst_stratum_coverage(result.covered, strata_test)),
            "stratified_coverage": {
                str(k): float(v)
                for k, v in stratified_coverage(result.covered, strata_test).items()
            },
            "mean_radius": float(mean_radius(result.radius)),
            "sscv": float(size_stratified_coverage_violation(result.covered, result.radius, alpha)),
            "coverage_variance": float(coverage_variance(result.covered, strata_test)),
            "runtime_sec": float(runtime_sec),
            "radius_by_strata": {
                str(k): float(v)
                for k, v in radius_by_strata(result.radius, strata_test).items()
            },
        }

    return rep_results


def aggregate_reps(all_reps: list[dict]) -> dict:
    """Aggregate metrics across repetitions."""
    methods = all_reps[0].keys()
    agg = {}
    for method in methods:
        scalar_keys = [
            "marginal_coverage",
            "max_disparity",
            "worst_stratum_coverage",
            "mean_radius",
            "sscv",
            "coverage_variance",
            "runtime_sec",
        ]
        agg[method] = {}
        for key in scalar_keys:
            values = [rep[method][key] for rep in all_reps]
            agg[method][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        # Aggregate per-strata coverage (some reps may lack certain strata)
        all_strata_keys: set[str] = set()
        for rep in all_reps:
            all_strata_keys.update(rep[method]["stratified_coverage"].keys())
        agg[method]["stratified_coverage"] = {}
        for s in sorted(all_strata_keys):
            vals = [
                rep[method]["stratified_coverage"][s]
                for rep in all_reps
                if s in rep[method]["stratified_coverage"]
            ]
            if vals:
                agg[method]["stratified_coverage"][s] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_reps": len(vals),
                }

        # Aggregate per-strata radius
        all_radius_keys: set[str] = set()
        for rep in all_reps:
            all_radius_keys.update(rep[method]["radius_by_strata"].keys())
        agg[method]["radius_by_strata"] = {}
        for s in sorted(all_radius_keys):
            vals = [
                rep[method]["radius_by_strata"][s]
                for rep in all_reps
                if s in rep[method]["radius_by_strata"]
            ]
            if vals:
                agg[method]["radius_by_strata"][s] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_reps": len(vals),
                }

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]
    log.info(f"Running experiment: {exp_name}")

    # ── Step 1: Load PBMC3K reference ──
    log.info("Loading PBMC3K reference data...")
    adata = sc.datasets.pbmc3k()
    log.info(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    celltype_key = cfg["data"]["celltype_key"]
    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)

    if celltype_key not in adata.obs.columns:
        log.info(f"  '{celltype_key}' not found, adding via KMeans clustering...")
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        pca = PCA(n_components=30, random_state=42)
        X_pca = pca.fit_transform(expr)
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        adata.obs[celltype_key] = "ct_" + kmeans.fit_predict(X_pca).astype(str)
        log.info(f"  Created {adata.obs[celltype_key].nunique()} cell type clusters")

    cell_type_names = sorted(np.unique(adata.obs[celltype_key].values))
    gene_names = adata.var_names.tolist()
    labels = adata.obs[celltype_key].values

    # ── Step 2: Generate pseudo-bulk + deconvolve (ONCE) ──
    base_seed = cfg["seed"]
    log.info("Generating pseudo-bulk dataset (once)...")
    pb = generate_pseudobulk(
        expr=expr,
        labels=labels,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
        n_samples=cfg["data"]["n_samples"],
        cells_per_sample=cfg["data"]["cells_per_sample"],
        concentration=cfg["data"]["concentration"],
        noise_sd=cfg["data"]["noise_sd"],
        seed=base_seed,
    )
    log.info(f"  Generated {pb.bulk.shape[0]} samples, {len(cell_type_names)} types")

    log.info("Running NNLS deconvolution (once)...")
    U = nnls_deconv(pb.bulk, pb.signature)
    log.info(f"  Deconvolved {U.shape[0]} samples")

    R = aitchison_dist(pb.proportions, U)
    log.info(f"  Computed residuals: mean={R.mean():.3f}, std={R.std():.3f}")

    # ── Step 3: 200 reps = different random cal/test splits ──
    n_reps = cfg["conformal"]["n_reps"]
    log.info(f"Running {n_reps} conformal reps (split-only)...")
    fixed_labels = None
    if cfg["evaluation"].get("fixed_strata", True):
        fixed_labels = precompute_fixed_strata(
            U,
            cfg["evaluation"]["strata_method"],
            cfg["evaluation"]["n_strata"],
            seed=base_seed,
        )

    all_reps = []
    for i in range(n_reps):
        rep_results = run_one_rep(R, U, cfg, i, base_seed, fixed_labels=fixed_labels)
        all_reps.append(rep_results)
        if (i + 1) % 50 == 0:
            log.info(f"  Completed {i + 1}/{n_reps} reps")

    agg = aggregate_reps(all_reps)

    # ── Save results ──
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "aggregated": agg, "raw": all_reps}, f, indent=2)

    log.info(f"Results saved to {out_path}")
    for method, metrics in agg.items():
        cov = metrics["marginal_coverage"]
        disp = metrics["max_disparity"]
        log.info(f"  {method}: coverage={cov['mean']:.3f}±{cov['std']:.3f}, "
                 f"disparity={disp['mean']:.3f}±{disp['std']:.3f}")


if __name__ == "__main__":
    main()
