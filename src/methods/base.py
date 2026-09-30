"""Common types for conformal methods."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ConformalResult:
    """Standard output of any conformal method.
    
    covered: boolean array (n_test,), whether Y_i is in the prediction set
    radius: set size surrogate (n_test,), local threshold * local scale
    threshold: the conformal quantile (scalar or per-group)
    """
    covered: np.ndarray
    radius: np.ndarray
    threshold: float | np.ndarray
