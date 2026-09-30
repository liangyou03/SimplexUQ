"""Oracle local conformal: normalize by known true σ(u)."""
import numpy as np
from .base import ConformalResult


def oracle_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    sigma_cal: np.ndarray,
    sigma_test: np.ndarray,
) -> ConformalResult:
    """Conformal prediction with oracle (known) local scale.

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        sigma_cal: true scale at calibration points (n_cal,)
        sigma_test: true scale at test points (n_test,)

    Returns:
        ConformalResult with locally-scaled radius.
    """
    S_cal = R_cal / sigma_cal
    n = len(S_cal)
    q = np.quantile(S_cal, np.ceil((1 - alpha) * (n + 1)) / n, method="higher")
    radius = sigma_test * q
    covered = R_test <= radius
    return ConformalResult(covered=covered, radius=radius, threshold=q)
