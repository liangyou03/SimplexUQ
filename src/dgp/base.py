"""Base DGP interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class DGPSample:
    """Standard output of any DGP.
    
    X: covariates (n, d)
    Y: true simplex responses (n, K)
    U: predictions f̂(X) (n, K)
    R: residuals d(Y, U) (n,)
    sigma_true: oracle local scale at U (n,), None if unavailable
    """
    X: np.ndarray
    Y: np.ndarray
    U: np.ndarray
    R: np.ndarray
    sigma_true: np.ndarray | None = None


class BaseDGP(ABC):
    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        ...
