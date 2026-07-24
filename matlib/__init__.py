from .rng import normal
from .linalg import transpose, permute, tensordot, matvec, matmul, outer
from .broadcasting import broadcast, elementwise
from .stats import (
    sum, mean, std, variance, max,
    sum_axis, mean_axis, variance_axis, max_axis,
)
from .ops import exp, log, sqrt, abs
from .manipulation import concatenate, stack, split, unbind

__all__ = [
    # rng
    "normal",
    # linalg
    "transpose",
    "permute",
    "tensordot",
    "matvec",
    "matmul",
    "outer",
    # broadcasting
    "broadcast",
    "elementwise",
    # stats – global
    "sum",
    "mean",
    "std",
    "variance",
    "max",
    # stats – axis-aware
    "sum_axis",
    "mean_axis",
    "variance_axis",
    "max_axis",
    # element-wise unary
    "exp",
    "log",
    "sqrt",
    "abs",
    # manipulation
    "concatenate",
    "stack",
    "split",
    "unbind",
]
