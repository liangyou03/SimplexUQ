"""Response-free overlapping group families for multivalid evaluation."""
from dataclasses import dataclass

import numpy as np

from .strata import precompute_fixed_strata


@dataclass(frozen=True)
class MultivalidGroups:
    """Fixed group membership matrix and aligned human-readable names."""

    membership: np.ndarray
    names: tuple[str, ...]


def _partition_columns(labels: np.ndarray, prefix: str) -> list[tuple[str, np.ndarray]]:
    """Expand integer partition labels into named Boolean columns."""
    return [
        (f"{prefix}={int(label)}", labels == label)
        for label in np.unique(labels)
    ]


def build_multivalid_groups(
    U: np.ndarray,
    default_labels: np.ndarray,
    n_bins: int = 5,
    dominant_groups: int = 5,
    min_group_fraction: float = 0.05,
    seed: int = 2026,
) -> MultivalidGroups:
    """Build a fixed overlapping family from prediction-space partitions.

    The family contains the universal group and one-hot columns for the task's
    default strata, entropy bins, boundary bins, and grouped dominant simplex
    components. Duplicate and undersupported columns are removed without using
    target values or residuals.

    Args:
        U: Frozen simplex prediction vectors with shape ``(n, K)``.
        default_labels: Pre-declared evaluation labels for the same rows.
        n_bins: Number of entropy and boundary bins.
        dominant_groups: Number of grouped dominant-component strata.
        min_group_fraction: Minimum full-sample support for a retained group.
        seed: Fixed seed passed to prediction-space stratification helpers.

    Returns:
        Deduplicated Boolean membership matrix and column names.
    """
    U = np.asarray(U, dtype=float)
    default_labels = np.asarray(default_labels)
    if U.ndim != 2 or len(U) == 0:
        raise ValueError("U must be a nonempty two-dimensional array")
    if default_labels.ndim != 1 or len(default_labels) != len(U):
        raise ValueError("default_labels must align with U")
    if not 0.0 <= min_group_fraction <= 1.0:
        raise ValueError("min_group_fraction must lie in [0, 1]")

    candidates: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(U), dtype=bool))
    ]
    candidates.extend(_partition_columns(default_labels, "default"))
    for method, count in (
        ("entropy", n_bins),
        ("boundary", n_bins),
        ("dominant", dominant_groups),
    ):
        labels = precompute_fixed_strata(U, method, count, seed=seed)
        candidates.extend(_partition_columns(labels, method))

    minimum_support = max(1, int(np.ceil(min_group_fraction * len(U))))
    retained_names: list[str] = []
    retained_columns: list[np.ndarray] = []
    seen: set[bytes] = set()
    for name, column in candidates:
        column = np.asarray(column, dtype=bool)
        key = column.tobytes()
        if int(column.sum()) < minimum_support or key in seen:
            continue
        seen.add(key)
        retained_names.append(name)
        retained_columns.append(column)

    if not retained_columns:
        raise ValueError("group filtering removed every group")
    return MultivalidGroups(
        membership=np.column_stack(retained_columns),
        names=tuple(retained_names),
    )
