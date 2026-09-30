"""kNN-based local scale estimation shared by local conformal methods."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..utils.simplex import ilr


def knn_sigma_hat(
    U_ref: np.ndarray,
    R_ref: np.ndarray,
    U_query: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Estimate local scale at query points via kNN in ILR space.

    Args:
        U_ref: reference simplex predictions (n_ref, K)
        R_ref: residuals at reference points (n_ref,)
        U_query: query simplex predictions (n_query, K)
        k: number of neighbors

    Returns:
        Estimated local scale (n_query,), floored at 1e-8.
    """
    if len(U_ref) == 0:
        return np.ones(len(U_query), dtype=float)
    Z_ref = ilr(U_ref)
    Z_query = ilr(U_query)
    k_actual = min(k, len(Z_ref))
    nn = NearestNeighbors(n_neighbors=k_actual).fit(Z_ref)
    _, indices = nn.kneighbors(Z_query)
    neighbor_R = R_ref[indices]  # (n_query, k_actual)
    sigma_hat = np.median(neighbor_R, axis=1)
    return np.maximum(sigma_hat, 1e-8)


def knn_sigma_leave_one_out(
    U_ref: np.ndarray,
    R_ref: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Estimate local scale at each reference point excluding itself.

    Args:
        U_ref: reference simplex predictions (n_ref, K)
        R_ref: residuals at reference points (n_ref,)
        k: number of non-self neighbors

    Returns:
        Leave-one-out local scales (n_ref,), floored at 1e-8.
    """
    if len(U_ref) <= 1:
        return np.ones(len(U_ref), dtype=float)

    Z_ref = ilr(U_ref)
    k_actual = min(k + 1, len(Z_ref))
    nn = NearestNeighbors(n_neighbors=k_actual).fit(Z_ref)
    _, indices = nn.kneighbors(Z_ref)

    loo_neighbors = indices[:, 1:] if k_actual > 1 else indices
    neighbor_R = R_ref[loo_neighbors]
    sigma_hat = np.median(neighbor_R, axis=1)
    return np.maximum(sigma_hat, 1e-8)
