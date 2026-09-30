"""Partition (grouped) conformal prediction."""
import numpy as np
from .base import ConformalResult
from ._split_quantile import split_conformal_quantile


def partition_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    strata_cal: np.ndarray,
    strata_test: np.ndarray,
) -> ConformalResult:
    """Grouped conformal with per-stratum thresholds.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        strata_cal: integer group labels for cal (n_cal,)
        strata_test: integer group labels for test (n_test,)

    Returns:
        ConformalResult with per-group coverage guarantee.
    """
    covered = np.zeros(len(R_test), dtype=bool)
    radius = np.full(len(R_test), np.inf)
    thresholds = {}

    for s in np.unique(strata_test):
        cal_mask = strata_cal == s
        test_mask = strata_test == s
        R_cal_s = R_cal[cal_mask]
        n_s = len(R_cal_s)

        if n_s == 0:
            # No cal points in this group — cover conservatively
            covered[test_mask] = True
            thresholds[int(s)] = np.inf
            continue

        q_s = split_conformal_quantile(R_cal_s, alpha)
        covered[test_mask] = R_test[test_mask] <= q_s
        radius[test_mask] = q_s
        thresholds[int(s)] = float(q_s)

    return ConformalResult(covered=covered, radius=radius, threshold=thresholds)
