"""Synthetic DGP registry exports.

Keep this module lightweight: real-data helpers such as pseudo-bulk generation
and NNLS deconvolution should be imported from their submodules directly so
synthetic experiments do not pay the import cost of heavy optional stacks.
"""

from .discrete_groups import DiscreteGroupsDGP
from .heavy_tail import HeavyTailDGP
from .high_k import HighKDGP
from .model_bias import ModelBiasDGP
from .pure_scale import PureScaleDGP
