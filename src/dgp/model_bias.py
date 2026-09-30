"""Model-bias DGP: predictor misspecification beyond pure local scale."""
import numpy as np
from .base import BaseDGP, DGPSample
from .pure_scale import PureScaleDGP
from ..utils.simplex import ilr, ilr_inv, entropy, aitchison_dist


class ModelBiasDGP(PureScaleDGP):
    """Predictor bias that cannot be summarized by a scalar local scale alone.

    The data-generating truth still follows the pure-scale construction, but the
    predictor is shifted in ILR space by a location-dependent bias field.
    Depending on `bias_type`, this field can rotate with prediction-space angle
    or vary smoothly with the covariates. This creates residual distributions
    whose heterogeneity is not purely radial.
    """
    def __init__(self, K: int = 3, sigma_min: float = 0.1, c: float = 0.5,
                 d_x: int = 2, bias_scale: float = 0.15, bias_type: str = "smooth"):
        super().__init__(K=K, sigma_min=sigma_min, c=c, d_x=d_x)
        self.bias_scale = bias_scale
        self.bias_type = bias_type
        self._B = None  # linear bias weights

    def _init_bias(self, rng: np.random.Generator):
        if self._B is None:
            self._B = rng.standard_normal((self.d_x, self.K - 1)) * 0.5

    def _bias(self, X: np.ndarray, Z_mu: np.ndarray) -> np.ndarray:
        """Location-dependent predictor bias in ILR space."""
        if self.bias_type == "linear":
            return self.bias_scale * (X @ self._B)

        if self.bias_type == "smooth":
            return self.bias_scale * np.sin(X @ self._B)

        if self.bias_type == "rotational":
            bias = np.zeros_like(Z_mu)
            z0 = Z_mu[:, 0]
            z1 = Z_mu[:, 1] if Z_mu.shape[1] > 1 else np.zeros_like(z0)
            phase = np.arctan2(z1, z0 + 1e-8)
            radius = np.sqrt(z0 ** 2 + z1 ** 2)
            amp = self.bias_scale * (0.5 + np.tanh(radius))
            bias[:, 0] = amp * np.cos(2.0 * phase)
            if Z_mu.shape[1] > 1:
                bias[:, 1] = amp * np.sin(2.0 * phase)
            if Z_mu.shape[1] > 2:
                bias[:, 2:] = 0.35 * self.bias_scale * np.sin(1.7 * Z_mu[:, 2:])
            return bias

        raise ValueError(f"Unknown bias_type: {self.bias_type}")

    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        self._init_weights(rng)
        self._init_bias(rng)

        X = rng.standard_normal((n, self.d_x))
        mu = self._mu(X)
        sigma = self._sigma(mu)

        # True Y: same as DGP 1
        Z_mu = ilr(mu)
        eps = rng.standard_normal((n, self.K - 1))
        Z_y = Z_mu + sigma[:, None] * eps
        Y = ilr_inv(Z_y, K=self.K)

        # Biased predictor
        b = self._bias(X, Z_mu)
        U = ilr_inv(Z_mu + b, K=self.K)

        R = aitchison_dist(Y, U)

        return DGPSample(X=X, Y=Y, U=U, R=R, sigma_true=sigma)
