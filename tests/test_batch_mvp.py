"""Tests for batch multivalid conformal prediction."""

import numpy as np
import pytest

from src.methods.batch_mvp import (
    _score_to_unit,
    _unit_to_score,
    batch_mvp_conformal,
)
from src.utils.multivalid_groups import build_multivalid_groups


def test_score_transform_round_trip():
    scores = np.array([0.0, 0.1, 1.0, 10.0, 100.0])

    transformed = _score_to_unit(scores)
    recovered = _unit_to_score(transformed)

    assert np.all((0.0 <= transformed) & (transformed < 1.0))
    np.testing.assert_allclose(recovered, scores)


def test_batch_mvp_rejects_mismatched_group_columns():
    with pytest.raises(ValueError, match="same number of columns"):
        batch_mvp_conformal(
            R_cal=np.array([0.1, 0.2]),
            R_test=np.array([0.15]),
            alpha=0.1,
            groups_cal=np.ones((2, 2), dtype=bool),
            groups_test=np.ones((1, 3), dtype=bool),
        )


def test_batch_mvp_returns_test_specific_radii_and_diagnostics():
    rng = np.random.default_rng(7)
    n_cal = 1000
    n_test = 2000
    cal_labels = np.repeat([0, 1], n_cal // 2)
    test_labels = np.repeat([0, 1], n_test // 2)
    R_cal = np.concatenate(
        [rng.uniform(0.0, 0.25, n_cal // 2), rng.uniform(0.5, 1.0, n_cal // 2)]
    )
    R_test = np.concatenate(
        [rng.uniform(0.0, 0.25, n_test // 2), rng.uniform(0.5, 1.0, n_test // 2)]
    )
    groups_cal = np.column_stack(
        [np.ones(n_cal, dtype=bool), cal_labels == 0, cal_labels == 1]
    )
    groups_test = np.column_stack(
        [np.ones(n_test, dtype=bool), test_labels == 0, test_labels == 1]
    )

    result = batch_mvp_conformal(
        R_cal,
        R_test,
        alpha=0.1,
        groups_cal=groups_cal,
        groups_test=groups_test,
        num_grid=100,
        max_rounds=200,
    )

    assert result.radius.shape == (n_test,)
    assert result.covered.shape == (n_test,)
    assert result.covered.dtype == bool
    assert result.n_groups == 3
    assert result.n_patches > 0
    assert np.unique(result.radius[test_labels == 0]).size == 1
    assert np.unique(result.radius[test_labels == 1]).size == 1
    assert result.radius[test_labels == 0][0] < result.radius[test_labels == 1][0]

    group_coverages = [result.covered[test_labels == group].mean() for group in (0, 1)]
    assert min(group_coverages) >= 0.86
    assert max(abs(coverage - 0.9) for coverage in group_coverages) <= 0.06


def test_batch_mvp_is_deterministic():
    R_cal = np.linspace(0.01, 1.0, 200)
    R_test = np.linspace(0.01, 1.0, 100)
    cal_half = np.arange(len(R_cal)) < len(R_cal) // 2
    test_half = np.arange(len(R_test)) < len(R_test) // 2
    groups_cal = np.column_stack([np.ones(len(R_cal), dtype=bool), cal_half, ~cal_half])
    groups_test = np.column_stack([np.ones(len(R_test), dtype=bool), test_half, ~test_half])

    first = batch_mvp_conformal(R_cal, R_test, 0.1, groups_cal, groups_test)
    second = batch_mvp_conformal(R_cal, R_test, 0.1, groups_cal, groups_test)

    np.testing.assert_array_equal(first.radius, second.radius)
    np.testing.assert_array_equal(first.covered, second.covered)
    assert first.patches == second.patches


def test_multivalid_groups_are_fixed_deduplicated_and_supported():
    rng = np.random.default_rng(11)
    U = rng.dirichlet(np.ones(4), size=400)
    default_labels = np.repeat(np.arange(4), 100)

    first = build_multivalid_groups(
        U,
        default_labels,
        n_bins=4,
        dominant_groups=4,
        min_group_fraction=0.05,
    )
    second = build_multivalid_groups(
        U,
        default_labels,
        n_bins=4,
        dominant_groups=4,
        min_group_fraction=0.05,
    )

    np.testing.assert_array_equal(first.membership, second.membership)
    assert first.names == second.names
    assert first.names[0] == "all"
    assert any(name.startswith("default=") for name in first.names)
    assert np.all(first.membership.sum(axis=0) >= 20)
    assert len({column.tobytes() for column in first.membership.T}) == len(first.names)


def test_multivalid_groups_filter_tiny_default_group():
    rng = np.random.default_rng(12)
    U = rng.dirichlet(np.ones(3), size=100)
    labels = np.zeros(100, dtype=int)
    labels[-1] = 1

    groups = build_multivalid_groups(U, labels, min_group_fraction=0.05)

    assert "default=1" not in groups.names
