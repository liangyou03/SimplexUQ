"""Two-stage split conformal with local normalization (Theorem 4.2)."""
import numpy as np
from .base import ConformalResult
from ._split_quantile import split_conformal_quantile
from ._knn_sigma import knn_sigma_hat


def twostage_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    U_cal: np.ndarray,
    U_test: np.ndarray,
    n_scale_est: int | None = None,
    k: int = 20,
) -> ConformalResult:
    """Two-stage locally-normalized conformal prediction.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        U_cal: calibration predictions (n_cal, K)
        U_test: test predictions (n_test, K)
        n_scale_est: points reserved for scale estimation (default: n_cal // 2)
        k: kNN neighbors for scale estimation

    Returns:
        ConformalResult with exact marginal coverage guarantee.
    """
    n_cal = len(R_cal)
    if n_cal < 2:
        radius = np.full(len(R_test), np.inf, dtype=float)
        return ConformalResult(covered=np.ones(len(R_test), dtype=bool), radius=radius, threshold=np.inf)
    if n_scale_est is None:
        n_scale_est = n_cal // 2
    if n_scale_est <= 0 or n_scale_est >= n_cal:
        raise ValueError("n_scale_est must leave at least one scale-estimation and one calibration point")

    # Stage 1: estimate σ̂ from first split
    U_est, R_est = U_cal[:n_scale_est], R_cal[:n_scale_est]
    U_cal2, R_cal2 = U_cal[n_scale_est:], R_cal[n_scale_est:]

    sigma_hat_cal2 = knn_sigma_hat(U_est, R_est, U_cal2, k=k)

    # Stage 2: calibrate on normalized scores
    S_cal2 = R_cal2 / sigma_hat_cal2
    q = split_conformal_quantile(S_cal2, alpha)

    # Test inference
    sigma_hat_test = knn_sigma_hat(U_est, R_est, U_test, k=k)
    radius = sigma_hat_test * q
    covered = R_test <= radius

    return ConformalResult(covered=covered, radius=radius, threshold=q)
