"""Size-stratified coverage metrics."""
import numpy as np


def size_stratified_coverage(
    covered: np.ndarray,
    radius: np.ndarray,
    n_bins: int = 5,
) -> dict[int, float]:
    """Coverage within radius/size bins.

    Bins are quantile-based to avoid empty tails when set sizes are skewed.
    """
    covered = np.asarray(covered, dtype=bool)
    radius = np.asarray(radius, dtype=float)

    if len(radius) == 0:
        return {}

    if np.allclose(radius, radius[0]):
        return {0: float(covered.mean())}

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(radius, quantiles))
    if len(edges) <= 1:
        return {0: float(covered.mean())}

    bin_ids = np.digitize(radius, edges[1:-1], right=True)
    result = {}
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        if mask.any():
            result[int(b)] = float(covered[mask].mean())
    return result


def size_stratified_coverage_violation(
    covered: np.ndarray,
    radius: np.ndarray,
    alpha: float,
    n_bins: int = 5,
) -> float:
    """Maximum coverage deviation across prediction-size bins."""
    sc = size_stratified_coverage(covered, radius, n_bins=n_bins)
    target = 1.0 - alpha
    return max(abs(v - target) for v in sc.values())
