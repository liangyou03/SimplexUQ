"""Pure-scale heterogeneity with heavy-tailed ILR noise."""
import numpy as np

from .base import DGPSample
from .pure_scale import PureScaleDGP
from ..utils.simplex import aitchison_dist, ilr, ilr_inv


class HeavyTailDGP(PureScaleDGP):
    """D2 with Student-t noise in ILR space."""

    def __init__(
        self,
        K: int = 3,
        sigma_min: float = 0.1,
        c: float = 0.5,
        d_x: int = 2,
        df: float = 3.0,
    ):
        super().__init__(K=K, sigma_min=sigma_min, c=c, d_x=d_x)
        self.df = df

    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        self._init_weights(rng)
        X = rng.standard_normal((n, self.d_x))
        mu = self._mu(X)
        sigma = self._sigma(mu)

        Z_mu = ilr(mu)
        scale = np.sqrt(self.df / (self.df - 2.0)) if self.df > 2 else 1.0
        eps = rng.standard_t(df=self.df, size=(n, self.K - 1)) / scale
        Y = ilr_inv(Z_mu + sigma[:, None] * eps, K=self.K)

        U = mu
        R = aitchison_dist(Y, U)
        return DGPSample(X=X, Y=Y, U=U, R=R, sigma_true=sigma)
