"""Shared split-conformal quantile utilities."""
import numpy as np


def split_conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Return the standard split-conformal threshold.

    The conformal order statistic is
    ``ceil((1 - alpha) * (n + 1))`` among ``n`` calibration scores. If that
    order statistic is beyond the calibration sample, the finite-sample
    conformal set is conservatively infinite.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.inf
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    return float(np.quantile(values, k / n, method="higher"))
