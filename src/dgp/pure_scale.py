"""DGP 1: Pure scale heterogeneity in ILR space.

Y = ilr⁻¹(ilr(μ(X)) + σ(μ(X))·ε),  ε ~ N(0, I)
σ(u) = σ_min + c·(1 - H(u)/log K)
f̂(X) = μ(X)
"""
import numpy as np
from .base import BaseDGP, DGPSample
from ..utils.simplex import ilr, ilr_inv, entropy, aitchison_dist


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class PureScaleDGP(BaseDGP):
    """Pure scale heterogeneity: residual distribution is N(0, σ²(u)I) in ILR space.

    Args:
        K: number of simplex components
        sigma_min: minimum scale
        c: scale range (σ_max = σ_min + c)
        d_x: dimension of covariate X
    """
    def __init__(self, K: int = 3, sigma_min: float = 0.1, c: float = 0.5,
                 d_x: int = 2):
        self.K = K
        self.sigma_min = sigma_min
        self.c = c
        self.d_x = d_x
        # Fixed linear map X -> logits, generated once
        self._W = None

    def _init_weights(self, rng: np.random.Generator):
        """Initialize fixed linear map for μ(X)."""
        if self._W is None:
            self._W = rng.standard_normal((self.d_x, self.K)) * 0.8

    def _mu(self, X: np.ndarray) -> np.ndarray:
        """Map X -> simplex via softmax of linear functions."""
        logits = X @ self._W
        return _softmax(logits)

    def _sigma(self, u: np.ndarray) -> np.ndarray:
        """Scale function: low entropy -> high scale."""
        H = entropy(u)
        return self.sigma_min + self.c * (1.0 - H / np.log(self.K))

    def sample(self, n: int, rng: np.random.Generator) -> DGPSample:
        self._init_weights(rng)

        X = rng.standard_normal((n, self.d_x))
        mu = self._mu(X)
        sigma = self._sigma(mu)

        # Generate Y in ILR space
        Z_mu = ilr(mu)
        eps = rng.standard_normal((n, self.K - 1))
        Z_y = Z_mu + sigma[:, None] * eps
        Y = ilr_inv(Z_y, K=self.K)

        # Predictor = oracle mean
        U = mu
        R = aitchison_dist(Y, U)

        return DGPSample(X=X, Y=Y, U=U, R=R, sigma_true=sigma)
