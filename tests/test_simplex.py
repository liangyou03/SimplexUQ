"""Tests for simplex utilities."""
import numpy as np
from src.utils.simplex import ilr, ilr_inv, entropy, aitchison_dist


def test_entropy_uniform():
    K = 3
    p = np.array([1/K] * K)
    assert np.isclose(entropy(p), np.log(K))


def test_entropy_degenerate():
    p = np.array([1.0, 0.0, 0.0])
    assert np.isclose(entropy(p), 0.0, atol=1e-10)


def test_ilr_shape():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    z = ilr(p)
    assert z.shape == (2, 2)


def test_aitchison_self_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert np.isclose(aitchison_dist(p, p), 0.0)


def test_ilr_inv_roundtrip_single():
    p = np.array([0.2, 0.3, 0.5])
    assert np.allclose(ilr_inv(ilr(p)), p, atol=1e-12)


def test_ilr_inv_roundtrip_batch():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert np.allclose(ilr_inv(ilr(p)), p, atol=1e-12)


def test_ilr_inv_roundtrip_k5():
    p = np.array([0.1, 0.15, 0.25, 0.3, 0.2])
    assert np.allclose(ilr_inv(ilr(p)), p, atol=1e-12)


def test_ilr_inv_valid_simplex():
    coords = np.array([0.5, -0.3])
    p = ilr_inv(coords)
    assert p.shape[-1] == 3
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_ilr_inv_zero_is_uniform():
    coords = np.zeros(3)  # K-1=3 -> K=4
    p = ilr_inv(coords)
    assert np.allclose(p, 0.25, atol=1e-12)
