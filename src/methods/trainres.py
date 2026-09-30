"""Local scaling from training residuals (expected to fail)."""
import numpy as np
from .base import ConformalResult
from ._split_quantile import split_conformal_quantile
from ._knn_sigma import knn_sigma_hat


def trainres_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    U_cal: np.ndarray,
    U_test: np.ndarray,
    R_train: np.ndarray,
    U_train: np.ndarray,
    k: int = 20,
) -> ConformalResult:
    """Locally-normalized conformal using training residuals for scale.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        U_cal: calibration predictions (n_cal, K)
        U_test: test predictions (n_test, K)
        R_train: training residuals (n_train,)
        U_train: training predictions (n_train, K)
        k: kNN neighbors for scale estimation

    Returns:
        ConformalResult (expected to be miscalibrated).
    """
    sigma_hat_cal = knn_sigma_hat(U_train, R_train, U_cal, k=k)
    S_cal = R_cal / sigma_hat_cal
    q = split_conformal_quantile(S_cal, alpha)

    sigma_hat_test = knn_sigma_hat(U_train, R_train, U_test, k=k)
    radius = sigma_hat_test * q
    covered = R_test <= radius

    return ConformalResult(covered=covered, radius=radius, threshold=q)
