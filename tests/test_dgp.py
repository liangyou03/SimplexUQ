"""Tests for synthetic DGP registry additions."""
import numpy as np

from src.dgp.discrete_groups import DiscreteGroupsDGP
from src.dgp.heavy_tail import HeavyTailDGP
from src.dgp.high_k import HighKDGP


def test_discrete_groups_shapes():
    rng = np.random.default_rng(0)
    sample = DiscreteGroupsDGP().sample(32, rng)
    assert sample.Y.shape == (32, 10)
    assert sample.U.shape == (32, 10)
    assert sample.R.shape == (32,)
    assert sample.sigma_true.shape == (32,)


def test_heavy_tail_shapes():
    rng = np.random.default_rng(0)
    sample = HeavyTailDGP().sample(16, rng)
    assert sample.Y.shape == (16, 3)
    assert sample.U.shape == (16, 3)
    assert sample.R.shape == (16,)


def test_high_k_shapes():
    rng = np.random.default_rng(0)
    sample = HighKDGP().sample(8, rng)
    assert sample.Y.shape == (8, 50)
    assert sample.U.shape == (8, 50)
    assert sample.R.shape == (8,)
