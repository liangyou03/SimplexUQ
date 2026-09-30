"""Reproducibility utilities."""
import numpy as np


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
