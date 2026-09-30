"""Batch multivalid conformal prediction for fixed group families.

This module adapts the patching algorithm from Jung et al. (2023), "Batch
Multivalid Conformal Prediction", to the project's stateless method API.
"""
from dataclasses import dataclass

import numpy as np

from .base import ConformalResult


@dataclass
class BatchMVPResult(ConformalResult):
    """Conformal result with BatchMVP optimization diagnostics."""

    n_groups: int
    n_patches: int
    patches: tuple[tuple[int, int, int], ...]
    max_violation: float


def _score_to_unit(scores: np.ndarray) -> np.ndarray:
    """Map nonnegative scores monotonically from ``[0, inf)`` to ``[0, 1)``."""
    scores = np.asarray(scores, dtype=float)
    return scores / (1.0 + scores)


def _unit_to_score(values: np.ndarray) -> np.ndarray:
    """Invert :func:`_score_to_unit`, allowing a unit value of one."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / (1.0 - values)


def _validate_inputs(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    groups_cal: np.ndarray,
    groups_test: np.ndarray,
    num_grid: int,
    max_rounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize BatchMVP inputs."""
    R_cal = np.asarray(R_cal, dtype=float)
    R_test = np.asarray(R_test, dtype=float)
    groups_cal = np.asarray(groups_cal, dtype=bool)
    groups_test = np.asarray(groups_test, dtype=bool)

    if R_cal.ndim != 1 or R_test.ndim != 1:
        raise ValueError("R_cal and R_test must be one-dimensional")
    if len(R_cal) == 0:
        raise ValueError("R_cal must contain at least one score")
    if np.any(~np.isfinite(R_cal)) or np.any(~np.isfinite(R_test)):
        raise ValueError("scores must be finite")
    if np.any(R_cal < 0.0) or np.any(R_test < 0.0):
        raise ValueError("scores must be nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if groups_cal.ndim != 2 or groups_test.ndim != 2:
        raise ValueError("group matrices must be two-dimensional")
    if groups_cal.shape[0] != len(R_cal) or groups_test.shape[0] != len(R_test):
        raise ValueError("group matrix rows must match the corresponding scores")
    if groups_cal.shape[1] != groups_test.shape[1]:
        raise ValueError("calibration and test groups must have the same number of columns")
    if groups_cal.shape[1] == 0:
        raise ValueError("at least one group is required")
    if np.any(groups_cal.sum(axis=0) == 0):
        raise ValueError("every group must contain at least one calibration point")
    if num_grid < 2:
        raise ValueError("num_grid must be at least two")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    return R_cal, R_test, groups_cal, groups_test


def _nearest_grid_index(value: float, grid: np.ndarray) -> int:
    """Return the nearest midpoint-grid index with deterministic tie-breaking."""
    return int(np.argmin(np.abs(grid - value)))


def _violation_matrix(
    scores: np.ndarray,
    threshold_index: np.ndarray,
    groups: np.ndarray,
    grid: np.ndarray,
    target: float,
) -> np.ndarray:
    """Compute the BatchMVP squared calibration violation per group/value cell."""
    point_index, group_index = np.nonzero(groups)
    cell_index = group_index * len(grid) + threshold_index[point_index]
    n_cells = groups.shape[1] * len(grid)
    counts = np.bincount(cell_index, minlength=n_cells).astype(float)
    covered = scores[point_index] <= grid[threshold_index[point_index]]
    covered_counts = np.bincount(
        cell_index, weights=covered.astype(float), minlength=n_cells
    )
    empirical = np.full(n_cells, target, dtype=float)
    np.divide(covered_counts, counts, out=empirical, where=counts > 0)
    return (counts * np.square(target - empirical)).reshape(groups.shape[1], len(grid))


def batch_mvp_conformal(
    R_cal: np.ndarray,
    R_test: np.ndarray,
    alpha: float,
    groups_cal: np.ndarray,
    groups_test: np.ndarray,
    num_grid: int = 100,
    max_rounds: int = 1000,
) -> BatchMVPResult:
    """Calibrate test-specific radii over a fixed overlapping group family.

    Args:
        R_cal: Nonnegative calibration residuals.
        R_test: Nonnegative test residuals, used only to evaluate coverage.
        alpha: Nominal miscoverage level.
        groups_cal: Boolean calibration group-membership matrix.
        groups_test: Boolean test group-membership matrix with matching columns.
        num_grid: Number of midpoint thresholds on the transformed score scale.
        max_rounds: Maximum number of BatchMVP patch updates.

    Returns:
        Test coverage/radii and deterministic patch diagnostics.
    """
    R_cal, R_test, groups_cal, groups_test = _validate_inputs(
        R_cal, R_test, alpha, groups_cal, groups_test, num_grid, max_rounds
    )
    target = 1.0 - alpha
    grid = (np.arange(num_grid, dtype=float) + 0.5) / num_grid
    scores_cal = _score_to_unit(R_cal)

    # The reference BatchMVP implementation initializes every point at the
    # lowest grid value and learns all subsequent threshold patches.
    initial_index = 0
    threshold_index_cal = np.full(len(R_cal), initial_index, dtype=np.int32)
    patches: list[tuple[int, int, int]] = []

    for _ in range(max_rounds):
        violations = _violation_matrix(
            scores_cal, threshold_index_cal, groups_cal, grid, target
        )
        group_index, old_index = np.unravel_index(np.argmax(violations), violations.shape)
        patch_mask = groups_cal[:, group_index] & (threshold_index_cal == old_index)
        if not np.any(patch_mask):
            break
        patch_quantile = float(np.quantile(scores_cal[patch_mask], target))
        new_index = _nearest_grid_index(patch_quantile, grid)
        if new_index == old_index:
            break
        threshold_index_cal[patch_mask] = new_index
        patches.append((int(group_index), int(old_index), int(new_index)))

    threshold_index_test = np.full(len(R_test), initial_index, dtype=np.int32)
    for group_index, old_index, new_index in patches:
        patch_mask = groups_test[:, group_index] & (threshold_index_test == old_index)
        threshold_index_test[patch_mask] = new_index

    unit_radius = grid[threshold_index_test]
    radius = _unit_to_score(unit_radius)
    final_violations = _violation_matrix(
        scores_cal, threshold_index_cal, groups_cal, grid, target
    )
    return BatchMVPResult(
        covered=R_test <= radius,
        radius=radius,
        threshold=unit_radius,
        n_groups=groups_cal.shape[1],
        n_patches=len(patches),
        patches=tuple(patches),
        max_violation=float(np.max(final_violations)),
    )
