"""Tests for new conformal method variants."""
import numpy as np

from src.methods.fullcp import full_conformal
from src.methods.global_cp import global_split_conformal
from src.methods.jackknife_plus import jackknife_plus_conformal
from src.methods.twostage import twostage_conformal
from src.methods.weighted_cp import weighted_conformal


def test_weighted_conformal_returns_test_specific_radii():
    R_cal = np.array([0.1, 0.2, 0.3, 0.4])
    R_test = np.array([0.15, 0.35])
    weights_cal = np.array([1.0, 1.0, 2.0, 4.0])
    weights_test = np.array([0.5, 10.0])

    res = weighted_conformal(R_cal, R_test, alpha=0.1, weights_cal=weights_cal, weights_test=weights_test)

    assert res.radius.shape == (2,)
    assert np.isinf(res.radius[1])
    assert res.covered.dtype == bool


def test_jackknife_plus_localized_radius_shape():
    rng = np.random.default_rng(0)
    U_cal = rng.dirichlet(np.ones(3), size=30)
    U_test = rng.dirichlet(np.ones(3), size=7)
    R_cal = rng.uniform(0.05, 0.5, size=30)
    R_test = rng.uniform(0.05, 0.5, size=7)

    res = jackknife_plus_conformal(R_cal, R_test, alpha=0.1, U_cal=U_cal, U_test=U_test, k=5)

    assert res.radius.shape == (7,)
    assert res.covered.shape == (7,)
    assert np.all(res.radius > 0)


def test_split_quantile_returns_infinite_radius_when_order_statistic_exceeds_sample():
    res = global_split_conformal(np.array([0.1]), np.array([0.2, 0.05]), alpha=0.1)

    assert np.all(np.isinf(res.radius))
    assert np.all(res.covered)


def test_local_methods_are_conservative_with_too_little_calibration_data():
    U_cal = np.array([[0.2, 0.3, 0.5]])
    U_test = np.array([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1]])
    R_cal = np.array([0.1])
    R_test = np.array([0.2, 0.3])

    two_stage = twostage_conformal(R_cal, R_test, alpha=0.1, U_cal=U_cal, U_test=U_test)
    full = full_conformal(R_cal, R_test, alpha=0.1, U_cal=U_cal, U_test=U_test)

    assert np.all(np.isinf(two_stage.radius))
    assert np.all(two_stage.covered)
    assert np.all(np.isinf(full.radius))
    assert np.all(full.covered)


def test_weighted_conformal_rejects_zero_total_weight():
    R_cal = np.array([0.1, 0.2])
    R_test = np.array([0.15])

    try:
        weighted_conformal(R_cal, R_test, alpha=0.1, weights_cal=np.zeros(2))
    except ValueError as exc:
        assert "positive total mass" in str(exc)
    else:
        raise AssertionError("expected ValueError")
