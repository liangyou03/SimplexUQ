"""High-dimensional simplex DGP."""
from .pure_scale import PureScaleDGP


class HighKDGP(PureScaleDGP):
    """Pure-scale heterogeneity in a high-K simplex regime."""

    def __init__(
        self,
        K: int = 50,
        sigma_min: float = 0.05,
        c: float = 0.35,
        d_x: int = 10,
    ):
        super().__init__(K=K, sigma_min=sigma_min, c=c, d_x=d_x)
