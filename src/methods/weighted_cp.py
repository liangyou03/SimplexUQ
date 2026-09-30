"""Weighted conformal prediction under test-dependent importance weights."""
import numpy as np

from .base import ConformalResult


def _weighted_quantile_with_test_mass(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    test_weight: float,
) -> float:
    """Compute the weighted conformal threshold for one test point.

    The test point contributes mass at +inf, matching the weighted split
    conformal recipe used under covariate shift.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]

    total_weight = sorted_weights.sum() + test_weight
    cutoff = (1.0 - alpha) * total_weight
    csum = np.cumsum(sorted_weights)
    idx = np.searchsorted(csum, cutoff, side="left")
    if idx >= len(sorted_values):
        return np.inf
    return float(sorted_values[idx])


def weighted_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    weights_cal: np.ndarray,
    weights_test: np.ndarray | None = None,
) -> ConformalResult:
    """Weighted split conformal.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        weights_cal: nonnegative calibration weights
        weights_test: nonnegative test weights. If omitted, uses ones.

    Returns:
        ConformalResult with test-specific weighted thresholds.
    """
    weights_cal = np.asarray(weights_cal, dtype=float)
    if weights_cal.shape != np.asarray(R_cal).shape:
        raise ValueError("weights_cal must have the same shape as R_cal")
    if not np.all(np.isfinite(weights_cal)):
        raise ValueError("weights_cal must be finite")
    if np.any(weights_cal < 0):
        raise ValueError("weights_cal must be nonnegative")
    if float(weights_cal.sum()) <= 0.0:
        raise ValueError("weights_cal must have positive total mass")

    if weights_test is None:
        weights_test = np.ones(len(R_test), dtype=float)
    else:
        weights_test = np.asarray(weights_test, dtype=float)
        if weights_test.shape != np.asarray(R_test).shape:
            raise ValueError("weights_test must have the same shape as R_test")
        if not np.all(np.isfinite(weights_test)):
            raise ValueError("weights_test must be finite")
        if np.any(weights_test < 0):
            raise ValueError("weights_test must be nonnegative")

    radius = np.array([
        _weighted_quantile_with_test_mass(R_cal, weights_cal, alpha, w_t)
        for w_t in weights_test
    ])
    covered = R_test <= radius
    return ConformalResult(covered=covered, radius=radius, threshold=radius.copy())
