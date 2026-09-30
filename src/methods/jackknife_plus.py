"""Approximate jackknife+ for simplex-valued prediction tasks.

This implementation uses leave-one-out local score normalization on the
calibration sample as a practical surrogate when full model retraining is
unavailable. It is closer in spirit to jackknife+ than split CP, but it is
still an approximation in this project because the underlying predictor is not
re-fit for each leave-one-out fold.
"""
import numpy as np

from ._knn_sigma import knn_sigma_hat, knn_sigma_leave_one_out
from ._split_quantile import split_conformal_quantile
from .base import ConformalResult


def jackknife_plus_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    U_cal: np.ndarray | None = None,
    U_test: np.ndarray | None = None,
    loo_scores: np.ndarray | None = None,
    k: int = 20,
) -> ConformalResult:
    """Approximate jackknife+ using leave-one-out calibration scores.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        U_cal: calibration predictions (n_cal, K), optional
        U_test: test predictions (n_test, K), optional
        loo_scores: pre-computed leave-one-out scores, optional
        k: kNN neighbors for local scale estimation when U inputs are provided

    Returns:
        ConformalResult with a global or locally-rescaled radius.
    """
    if loo_scores is None:
        if U_cal is not None and U_test is not None:
            sigma_loo = knn_sigma_leave_one_out(U_cal, R_cal, k=k)
            loo_scores = R_cal / sigma_loo
        else:
            loo_scores = np.asarray(R_cal, dtype=float)
    else:
        loo_scores = np.asarray(loo_scores, dtype=float)

    q = split_conformal_quantile(loo_scores, alpha)

    if U_cal is not None and U_test is not None:
        sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=k)
        radius = sigma_test * q
    else:
        radius = np.full_like(R_test, q, dtype=float)

    covered = R_test <= radius
    return ConformalResult(covered=covered, radius=radius, threshold=q)
