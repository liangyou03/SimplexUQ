"""Coverage metrics."""
import numpy as np


def marginal_coverage(covered: np.ndarray) -> float:
    """Overall empirical coverage."""
    return covered.mean()


def stratified_coverage(
    covered: np.ndarray,
    strata: np.ndarray,
) -> dict[int, float]:
    """Stratified coverage in prediction space.
    
    Args:
        covered: boolean array (n,)
        strata: integer group labels (n,)
    
    Returns:
        dict mapping stratum label -> empirical coverage
    """
    result = {}
    for s in np.unique(strata):
        mask = strata == s
        if mask.sum() > 0:
            result[int(s)] = covered[mask].mean()
    return result


def max_disparity(
    covered: np.ndarray,
    strata: np.ndarray,
    alpha: float,
) -> float:
    """Max |stratified_coverage - (1-alpha)| across strata."""
    sc = stratified_coverage(covered, strata)
    target = 1.0 - alpha
    return max(abs(v - target) for v in sc.values())


def worst_stratum_coverage(
    covered: np.ndarray,
    strata: np.ndarray,
) -> float:
    """Minimum empirical coverage across strata."""
    sc = stratified_coverage(covered, strata)
    return min(sc.values())


def coverage_variance(
    covered: np.ndarray,
    strata: np.ndarray,
) -> float:
    """Variance of empirical coverage across strata."""
    sc = stratified_coverage(covered, strata)
    return float(np.var(list(sc.values())))
