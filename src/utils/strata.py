"""Prediction-space stratification utilities."""
import numpy as np
from .simplex import entropy, ilr
from sklearn.cluster import KMeans


def stratify_by_entropy(U: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin predictions by Shannon entropy."""
    H = entropy(U)
    return np.digitize(H, np.linspace(H.min(), H.max(), n_bins + 1)[1:-1])


def stratify_by_boundary(U: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin predictions by proximity to simplex boundary (min component)."""
    bprox = U.min(axis=-1)
    return np.digitize(bprox, np.linspace(bprox.min(), bprox.max(), n_bins + 1)[1:-1])


def stratify_by_kmeans(U: np.ndarray, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    """K-means clustering in ILR space."""
    Z = ilr(U)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(Z)


def stratify_by_argmax_group(U: np.ndarray, split_index: int = 5) -> np.ndarray:
    """Two-group stratification based on whether argmax(U) is before split_index."""
    top_class = np.argmax(U, axis=1)
    return (top_class >= split_index).astype(int)


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return stable interior quantile edges for binning a scalar score."""
    if n_bins <= 1:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, qs)
    return np.unique(edges)


def _digitize_fixed(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Digitize values using globally fixed quantile edges."""
    edges = _quantile_edges(values, n_bins)
    if edges.size == 0:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges)


def precompute_fixed_strata(
    U: np.ndarray,
    method: str,
    n_strata: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Precompute a fixed stratification on a full cached prediction matrix.

    This is intended for repeated cal/test splits of the same frozen task, where
    the stratification rule should remain constant across repetitions.

    Args:
        U: full prediction matrix with shape (n, K).
        method: one of {"entropy", "boundary", "dominant", "kmeans", "random"}.
        n_strata: target number of strata. For "dominant", this is treated as
            the number of grouped dominant-component bins; if it is at least K,
            each dominant component gets its own stratum.
        seed: random seed for methods that require stochastic initialization.

    Returns:
        Integer stratum labels of shape (n,).
    """
    method = method.lower()

    if method == "entropy":
        return _digitize_fixed(entropy(U), n_strata)

    if method == "boundary":
        return _digitize_fixed(U.min(axis=-1), n_strata)

    if method == "dominant":
        top = np.argmax(U, axis=1)
        k = U.shape[1]
        n_groups = min(max(int(n_strata), 1), k)
        if n_groups >= k:
            return top.astype(int)
        return np.floor(top * n_groups / k).astype(int)

    if method == "random":
        z = ilr(U)
        rng = np.random.default_rng(seed)
        direction = rng.normal(size=z.shape[1])
        direction /= np.linalg.norm(direction) + 1e-12
        score = z @ direction
        return _digitize_fixed(score, n_strata)

    if method == "kmeans":
        z = ilr(U)
        km = KMeans(n_clusters=n_strata, random_state=seed, n_init=10)
        return km.fit_predict(z)

    raise ValueError(f"Unknown fixed stratification method: {method}")
