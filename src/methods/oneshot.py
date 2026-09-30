"""One-shot kNN local scaling (no data splitting, no exchangeability guarantee)."""
import numpy as np
from .base import ConformalResult
from ._split_quantile import split_conformal_quantile
from ._knn_sigma import knn_sigma_hat


def oneshot_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    U_cal: np.ndarray,
    U_test: np.ndarray,
    k: int = 20,
) -> ConformalResult:
    """One-shot locally-normalized conformal (no split for scale estimation).

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        U_cal: calibration predictions (n_cal, K)
        U_test: test predictions (n_test, K)
        k: kNN neighbors for scale estimation

    Returns:
        ConformalResult (no formal coverage guarantee).
    """
    # Estimate σ̂ from same calibration set (breaks exchangeability)
    sigma_hat_cal = knn_sigma_hat(U_cal, R_cal, U_cal, k=k)
    S_cal = R_cal / sigma_hat_cal
    q = split_conformal_quantile(S_cal, alpha)

    sigma_hat_test = knn_sigma_hat(U_cal, R_cal, U_test, k=k)
    radius = sigma_hat_test * q
    covered = R_test <= radius

    return ConformalResult(covered=covered, radius=radius, threshold=q)
