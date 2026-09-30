"""Global split conformal prediction."""
import numpy as np
from .base import ConformalResult
from ._split_quantile import split_conformal_quantile


def global_split_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
) -> ConformalResult:
    """Standard split conformal with a single global threshold.
    
    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
    
    Returns:
        ConformalResult
    """
    q = split_conformal_quantile(R_cal, alpha)
    covered = R_test <= q
    radius = np.full_like(R_test, q, dtype=float)
    return ConformalResult(covered=covered, radius=radius, threshold=q)
