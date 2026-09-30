"""Exp 1.4: Training residuals collapse along KRR regularization path.

Varies KRR regularization λ. As λ→0:
- Training residuals shrink heterogeneously (dense regions → smaller)
- σ̂_train underestimates true σ, especially in dense regions
- trainres disparity worsens — can exceed global disparity
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.kernel_ridge import KernelRidge

from src.dgp import PureScaleDGP
from src.methods.trainres import trainres_conformal
from src.methods.global_cp import global_split_conformal
from src.methods.oracle import oracle_conformal
from src.metrics import marginal_coverage, max_disparity, stratified_coverage
from src.utils.seed import get_rng
from src.utils.simplex import ilr, ilr_inv, aitchison_dist
from src.utils.strata import stratify_by_boundary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_one_lambda(dgp, cfg: dict, lam: float, rng: np.random.Generator) -> dict:
    """Run one rep at a given regularization strength."""
    n_train = cfg["data"]["n_train"]
    n_cal = cfg["data"]["n_cal"]
    n_test = cfg["data"]["n_test"]
    alpha = cfg["evaluation"]["alpha"]

    total = n_train + n_cal + n_test
    sample = dgp.sample(total, rng)

    X_train = sample.X[:n_train]
    Y_train = sample.Y[:n_train]
    X_cal = sample.X[n_train:n_train + n_cal]
    Y_cal = sample.Y[n_train:n_train + n_cal]
    X_test = sample.X[n_train + n_cal:]
    Y_test = sample.Y[n_train + n_cal:]

    sigma_cal = sample.sigma_true[n_train:n_train + n_cal]
    sigma_test = sample.sigma_true[n_train + n_cal:]

    # KRR predictor in ILR space
    Z_train = ilr(Y_train)
    # gamma = 1 / (n_features * X.var()) mimics sklearn's "scale" heuristic
    gamma = 1.0 / (X_train.shape[1] * X_train.var())
    model = KernelRidge(alpha=lam, kernel="rbf", gamma=gamma)
    model.fit(X_train, Z_train)

    Z_train_hat = model.predict(X_train)
    Z_cal_hat = model.predict(X_cal)
    Z_test_hat = model.predict(X_test)

    U_train = ilr_inv(Z_train_hat)
    U_cal = ilr_inv(Z_cal_hat)
    U_test = ilr_inv(Z_test_hat)

    R_train = aitchison_dist(Y_train, U_train)
    R_cal = aitchison_dist(Y_cal, U_cal)
    R_test = aitchison_dist(Y_test, U_test)

    # Strata on oracle predictions for consistent stratification
    U_test_oracle = sample.U[n_train + n_cal:]
    strata_test = stratify_by_boundary(U_test_oracle, 5)

    # Three methods: trainres, global, oracle
    result_tr = trainres_conformal(R_cal, R_test, alpha, U_cal, U_test, R_train, U_train)
    result_gl = global_split_conformal(R_cal, R_test, alpha)
    result_or = oracle_conformal(R_cal, R_test, alpha, sigma_cal, sigma_test)

    strat_tr = {str(k): float(v) for k, v in stratified_coverage(result_tr.covered, strata_test).items()}

    return {
        "lambda": lam,
        "mean_R_train": float(R_train.mean()),
        "mean_R_cal": float(R_cal.mean()),
        "ratio_train_cal": float(R_train.mean() / max(R_cal.mean(), 1e-10)),
        "trainres": {
            "marginal_coverage": float(marginal_coverage(result_tr.covered)),
            "max_disparity": float(max_disparity(result_tr.covered, strata_test, alpha)),
            "stratified_coverage": strat_tr,
        },
        "global": {
            "marginal_coverage": float(marginal_coverage(result_gl.covered)),
            "max_disparity": float(max_disparity(result_gl.covered, strata_test, alpha)),
        },
        "oracle": {
            "marginal_coverage": float(marginal_coverage(result_or.covered)),
            "max_disparity": float(max_disparity(result_or.covered, strata_test, alpha)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]
    log.info(f"Running experiment: {exp_name}")

    dgp_cfg = cfg["dgp"]
    dgp = PureScaleDGP(K=dgp_cfg["K"], sigma_min=dgp_cfg["sigma_min"], c=dgp_cfg["c"])

    lambdas = cfg["trainres_path"]["values"]
    n_rep = cfg["data"]["n_rep"]
    base_seed = cfg["seed"]

    all_results = {}
    for lam in lambdas:
        log.info(f"  lambda={lam}")
        reps = []
        for i in range(n_rep):
            rng = get_rng(base_seed + i)
            reps.append(run_one_lambda(dgp, cfg, lam, rng))

        cov_tr = [r["trainres"]["marginal_coverage"] for r in reps]
        disp_tr = [r["trainres"]["max_disparity"] for r in reps]
        disp_gl = [r["global"]["max_disparity"] for r in reps]
        ratio = [r["ratio_train_cal"] for r in reps]

        all_results[str(lam)] = {
            "trainres_coverage": {"mean": float(np.mean(cov_tr)), "std": float(np.std(cov_tr))},
            "trainres_disparity": {"mean": float(np.mean(disp_tr)), "std": float(np.std(disp_tr))},
            "global_disparity": {"mean": float(np.mean(disp_gl)), "std": float(np.std(disp_gl))},
            "R_train_R_cal_ratio": {"mean": float(np.mean(ratio)), "std": float(np.std(ratio))},
            "raw": reps,
        }

        log.info(f"    trainres cov={np.mean(cov_tr):.3f}, disp={np.mean(disp_tr):.3f}, "
                 f"global disp={np.mean(disp_gl):.3f}, "
                 f"R_train/R_cal={np.mean(ratio):.3f}")

    # Save
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "results": all_results}, f, indent=2)

    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
