"""DGP with discrete heterogeneity aligned to coarse simplex groups."""
import numpy as np

from .base import BaseDGP, DGPSample
from .pure_scale import PureScaleDGP
from ..utils.simplex import aitchison_dist, ilr, ilr_inv


class DiscreteGroupsDGP(PureScaleDGP):
    """Step-function scale heterogeneity based on the predicted top class."""

    def __init__(
        self,
        K: int = 10,
        sigma_low: float = 0.08,
        sigma_high: float = 0.30,
        d_x: int = 5,
        easy_classes: int = 5,
    ):
        super().__init__(K=K, sigma_min=sigma_low, c=sigma_high - sigma_low, d_x=d_x)
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.easy_classes = easy_classes

    def _sigma(self, u: np.ndarray) -> np.ndarray:
        top_class = np.argmax(u, axis=1)
        is_easy = top_class < self.easy_classes
        return np.where(is_easy, self.sigma_low, self.sigma_high)

    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        self._init_weights(rng)
        X = rng.standard_normal((n, self.d_x))
        mu = self._mu(X)
        sigma = self._sigma(mu)

        Z_mu = ilr(mu)
        eps = rng.standard_normal((n, self.K - 1))
        Y = ilr_inv(Z_mu + sigma[:, None] * eps, K=self.K)
        U = mu
        R = aitchison_dist(Y, U)
        return DGPSample(X=X, Y=Y, U=U, R=R, sigma_true=sigma)
