"""Tests for coverage metrics."""
import numpy as np

from src.metrics.coverage import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.metrics.sscv import size_stratified_coverage, size_stratified_coverage_violation


def test_marginal_perfect():
    assert marginal_coverage(np.ones(100, dtype=bool)) == 1.0


def test_stratified_basic():
    covered = np.array([True, True, False, False])
    strata = np.array([0, 0, 1, 1])
    sc = stratified_coverage(covered, strata)
    assert sc[0] == 1.0
    assert sc[1] == 0.0


def test_max_disparity():
    covered = np.array([True, True, False, False])
    strata = np.array([0, 0, 1, 1])
    d = max_disparity(covered, strata, alpha=0.1)
    assert d == 0.9  # max(|1.0-0.9|, |0.0-0.9|) = 0.9


def test_worst_stratum_coverage():
    covered = np.array([True, True, False, False])
    strata = np.array([0, 0, 1, 1])
    assert worst_stratum_coverage(covered, strata) == 0.0


def test_coverage_variance():
    covered = np.array([True, True, False, False])
    strata = np.array([0, 0, 1, 1])
    assert np.isclose(coverage_variance(covered, strata), 0.25)


def test_sscv_single_bin_for_constant_radius():
    covered = np.array([True, True, False, True])
    radius = np.ones(4)
    sc = size_stratified_coverage(covered, radius, n_bins=5)
    assert sc == {0: 0.75}
    sscv = size_stratified_coverage_violation(covered, radius, alpha=0.1, n_bins=5)
    assert np.isclose(sscv, 0.15)
