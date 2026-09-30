"""Scale heterogeneity with mild tail/covariance drift."""
import numpy as np

from .base import DGPSample
from .pure_scale import PureScaleDGP
from ..utils.simplex import aitchison_dist, entropy, ilr, ilr_inv


class TailDriftDGP(PureScaleDGP):
    """Pure-scale DGP with entropy-dependent anisotropic ILR noise.

    This auxiliary regime is not part of the main table, but it is useful for
    stress-testing local normalization when the residual shape changes as well
    as the scalar residual scale.
    """

    def __init__(
        self,
        K: int = 3,
        sigma_min: float = 0.1,
        c: float = 0.5,
        d_x: int = 2,
        drift_strength: float = 0.5,
    ):
        super().__init__(K=K, sigma_min=sigma_min, c=c, d_x=d_x)
        self.drift_strength = drift_strength

    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        self._init_weights(rng)
        X = rng.standard_normal((n, self.d_x))
        mu = self._mu(X)
        sigma = self._sigma(mu)

        Z_mu = ilr(mu)
        H = entropy(mu) / np.log(self.K)
        eps = rng.standard_normal((n, self.K - 1))
        if self.K > 2:
            # Low-entropy predictions get a mild directional covariance tilt.
            eps[:, 0] *= 1.0 + self.drift_strength * (1.0 - H)
            eps[:, 1:] *= 1.0 - 0.5 * self.drift_strength * (1.0 - H[:, None])
        else:
            eps[:, 0] *= 1.0 + self.drift_strength * (1.0 - H)
        Y = ilr_inv(Z_mu + sigma[:, None] * eps, K=self.K)
        U = mu
        R = aitchison_dist(Y, U)
        return DGPSample(X=X, Y=Y, U=U, R=R, sigma_true=sigma)
