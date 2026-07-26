from .rng import normal
from .linalg import det, inverse, cross, solve, transpose, permute, tensordot, matvec, matmul, outer, cholesky
from .broadcasting import broadcast, elementwise
from .stats import (
    sum, mean, std, variance, max,
    sum_axis, mean_axis, variance_axis, max_axis,
)
from .ops import exp, log, sqrt, abs, tan, sin, cos, asin, acos, atan, sinh, cosh, tanh
from .manipulation import concatenate, stack, split, unbind, flip

__all__ = [
    # rng
    "normal",
    # linalg
    "det",
    "inverse",
    "transpose",
    "permute",
    "tensordot",
    "matvec",
    "matmul",
    "outer",
    "solve",
    "cross",
    "cholesky",
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
    "tan",
    "sin",
    "cos",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    # manipulation
    "concatenate",
    "stack",
    "split",
    "unbind",
    "flip"
]
