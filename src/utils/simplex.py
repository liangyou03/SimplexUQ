"""ILR transform and simplex geometry utilities."""
import numpy as np


def _helmert_matrix(K: int) -> np.ndarray:
    """Build (K-1, K) Helmert submatrix for ILR transform.

    Matches the forward ILR: coords[j] = sqrt((j+1)/(j+2)) * (mean(log_p[:j+1]) - log_p[j+1])
    """
    V = np.zeros((K - 1, K))
    for j in range(K - 1):
        V[j, :j + 1] = np.sqrt(1.0 / ((j + 1) * (j + 2)))
        V[j, j + 1] = -np.sqrt((j + 1) / (j + 2))
    return V


def ilr(p: np.ndarray) -> np.ndarray:
    """Isometric log-ratio transform. p: (..., K) -> (..., K-1)."""
    K = p.shape[-1]
    V = _helmert_matrix(K)
    log_p = np.log(np.clip(p, 1e-15, None))
    return log_p @ V.T  # (..., K) @ (K, K-1) -> (..., K-1)


def ilr_inv(coords: np.ndarray, K: int | None = None) -> np.ndarray:
    """Inverse ILR transform. coords: (..., K-1) -> (..., K).

    Args:
        coords: ILR coordinates (..., K-1)
        K: number of simplex components. If None, inferred as coords.shape[-1] + 1.

    Returns:
        Simplex vectors (..., K), rows sum to 1.
    """
    if K is None:
        K = coords.shape[-1] + 1
    V = _helmert_matrix(K)
    log_p = coords @ V  # (..., K-1) @ (K-1, K) -> (..., K)
    log_p -= log_p.max(axis=-1, keepdims=True)  # numerical stability
    p = np.exp(log_p)
    return p / p.sum(axis=-1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of simplex vectors. p: (..., K) -> (...)."""
    p_safe = np.clip(p, 1e-15, None)
    return -(p_safe * np.log(p_safe)).sum(axis=-1)


def aitchison_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Aitchison distance between simplex vectors."""
    d = ilr(p) - ilr(q)
    return np.sqrt((d ** 2).sum(axis=-1))
