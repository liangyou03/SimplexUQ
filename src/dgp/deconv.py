"""Bulk deconvolution via constrained least squares."""
import numpy as np
from scipy.optimize import nnls
import logging

log = logging.getLogger(__name__)


def nnls_deconv(bulk: np.ndarray, signature: np.ndarray) -> np.ndarray:
    """Deconvolve bulk expression into cell type proportions via NNLS.

    Solves: bulk_i ≈ signature @ p_i, s.t. p_i >= 0, then normalizes to simplex.

    Args:
        bulk: (n_samples, n_genes)
        signature: (n_genes, K)

    Returns:
        proportions_hat: (n_samples, K), rows sum to 1
    """
    n = bulk.shape[0]
    K = signature.shape[1]
    props = np.zeros((n, K))

    for i in range(n):
        coef, _ = nnls(signature, bulk[i])
        total = coef.sum()
        if total > 0:
            props[i] = coef / total
        else:
            props[i] = 1.0 / K  # fallback to uniform

    log.info(f"Deconvolved {n} samples into {K} types")
    return props
