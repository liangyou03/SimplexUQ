"""Symmetrized full conformal with local normalization (Theorem 4.1)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import ConformalResult
from ..utils.simplex import ilr


def full_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    U_cal: np.ndarray,
    U_test: np.ndarray,
    k: int = 20,
) -> ConformalResult:
    """Symmetrized full conformal prediction with local normalization.

    For each test point j, constructs augmented set = cal ∪ {j}, computes
    LOO sigma for all n_cal+1 points in the augmented set, then derives
    the conformal p-value. Exact per Theorem 4.1.

    Optimization: cal-point LOO sigmas are pre-computed on the cal set and
    updated only when the test point falls among a cal point's k nearest
    neighbors (rare for large n_cal).

    Args:
        R_cal: calibration residuals (n_cal,)
        R_test: test residuals (n_test,)
        alpha: miscoverage level
        U_cal: calibration predictions (n_cal, K)
        U_test: test predictions (n_test, K)
        k: kNN neighbors for leave-one-out scale estimation

    Returns:
        ConformalResult with exact marginal coverage guarantee.
    """
    n_cal = len(R_cal)
    n_test = len(R_test)
    if n_cal < 2:
        radius = np.full(n_test, np.inf, dtype=float)
        return ConformalResult(covered=np.ones(n_test, dtype=bool), radius=radius, threshold=alpha)
    k_loo = min(k, n_cal - 1)

    Z_cal = ilr(U_cal)
    Z_test = ilr(U_test)

    # Pre-compute cal LOO sigmas (reused across test points)
    nn_cal = NearestNeighbors(n_neighbors=k_loo + 1).fit(Z_cal)
    cal_dists, cal_nn_idx = nn_cal.kneighbors(Z_cal)
    cal_loo_sigma = np.zeros(n_cal)
    for i in range(n_cal):
        neighbor_idx = cal_nn_idx[i][cal_nn_idx[i] != i][:k_loo]
        cal_loo_sigma[i] = max(np.median(R_cal[neighbor_idx]), 1e-8)

    # Pre-compute cal normalized scores (base, without augmentation effect)
    S_cal_base = R_cal / cal_loo_sigma

    covered = np.zeros(n_test, dtype=bool)
    radius = np.zeros(n_test)

    for j in range(n_test):
        # For the test point: its LOO sigma is computed using cal as reference
        # (in the augmented set, dropping itself leaves exactly cal)
        test_dists, test_nn_idx = nn_cal.kneighbors(Z_test[j:j + 1])
        test_nbrs = test_nn_idx[0][:k_loo]
        sigma_test_j = max(np.median(R_cal[test_nbrs]), 1e-8)

        # For cal points: check if adding the test point changes their LOO sigma
        # A cal point's LOO sigma changes only if the test point is closer than
        # its k-th nearest cal neighbor. For efficiency, compute the updated
        # sigma only for affected cal points.
        S_cal = S_cal_base.copy()

        # Distance from each cal point to this test point
        d_to_test = np.linalg.norm(Z_cal - Z_test[j], axis=1)
        # k-th neighbor distance for each cal point (last column of cal_dists,
        # but we used k_loo+1 neighbors including self, so k-th non-self is index k_loo)
        kth_dist = cal_dists[:, k_loo]  # distance to (k_loo+1)-th neighbor (including self)

        affected = d_to_test < kth_dist
        if np.any(affected):
            for i in np.where(affected)[0]:
                # Recompute LOO sigma: drop self from cal, add test point
                orig_nbrs = cal_nn_idx[i][cal_nn_idx[i] != i][:k_loo]
                # Replace the farthest neighbor with test point if closer
                nbr_residuals = list(R_cal[orig_nbrs])
                # Drop the farthest cal neighbor, add test residual
                nbr_residuals[-1] = R_test[j]
                new_sigma = max(np.median(nbr_residuals), 1e-8)
                S_cal[i] = R_cal[i] / new_sigma

        S_test_j = R_test[j] / sigma_test_j

        # p-value: fraction of augmented scores >= test score
        n_geq = np.sum(S_cal >= S_test_j) + 1  # +1 for test point itself
        p_val = n_geq / (n_cal + 1)

        covered[j] = p_val > alpha
        # Radius: effective threshold
        all_S = np.concatenate([S_cal, [S_test_j]])
        q_idx = int(np.ceil((1 - alpha) * (n_cal + 1))) - 1
        radius[j] = sigma_test_j * np.sort(all_S)[min(q_idx, len(all_S) - 1)]

    return ConformalResult(covered=covered, radius=radius, threshold=alpha)
