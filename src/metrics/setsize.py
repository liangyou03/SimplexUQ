"""Set size metrics."""
import numpy as np


def mean_radius(radius: np.ndarray) -> float:
    return radius.mean()


def radius_by_strata(
    radius: np.ndarray,
    strata: np.ndarray,
) -> dict[int, float]:
    result = {}
    for s in np.unique(strata):
        mask = strata == s
        if mask.sum() > 0:
            result[int(s)] = radius[mask].mean()
    return result


def _distance_to_center(
    samples: np.ndarray,
    center: np.ndarray,
    score: str,
) -> np.ndarray:
    if score == "aitchison":
        from src.utils.simplex import aitchison_dist

        tiled_center = np.repeat(center[None, :], len(samples), axis=0)
        return aitchison_dist(samples, tiled_center)
    if score in {"tv", "total_variation", "l1"}:
        return 0.5 * np.sum(np.abs(samples - center[None, :]), axis=1)
    raise ValueError(f"Unsupported score for volume ratio: {score}")


def volume_ratio(
    center: np.ndarray,
    radius: float,
    *,
    score: str = "aitchison",
    n_mc: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo estimate of simplex volume covered by the score ball.

    The estimate is the fraction of uniformly sampled simplex points whose
    distance to `center` is at most `radius`.
    """
    center = np.asarray(center, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    samples = rng.dirichlet(np.ones(center.shape[-1]), size=n_mc)
    d = _distance_to_center(samples, center, score)
    return float(np.mean(d <= radius))


def mean_volume_ratio(
    centers: np.ndarray,
    radius: np.ndarray,
    *,
    score: str = "aitchison",
    n_mc: int = 20000,
    max_points: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Average Monte Carlo simplex-volume ratio across test points."""
    centers = np.asarray(centers, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)

    idx = np.arange(len(centers))
    if max_points is not None and len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    vals = [
        volume_ratio(centers[i], radius[i], score=score, n_mc=n_mc, rng=rng)
        for i in idx
    ]
    return float(np.mean(vals))


def volume_ratio_by_strata(
    centers: np.ndarray,
    radius: np.ndarray,
    strata: np.ndarray,
    *,
    score: str = "aitchison",
    n_mc: int = 20000,
    max_points: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    """Average simplex-volume ratio per prediction-space stratum."""
    centers = np.asarray(centers, dtype=float)
    radius = np.asarray(radius, dtype=float)
    strata = np.asarray(strata)
    if rng is None:
        rng = np.random.default_rng(0)

    result = {}
    for s in np.unique(strata):
        mask = strata == s
        centers_s = centers[mask]
        radius_s = radius[mask]
        result[int(s)] = mean_volume_ratio(
            centers_s,
            radius_s,
            score=score,
            n_mc=n_mc,
            max_points=max_points,
            rng=rng,
        )
    return result
