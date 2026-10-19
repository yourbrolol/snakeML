"""Math kernel library for snakeML.

Pure-Python array math over :mod:`structs` primitives: broadcasting and
element-wise ops (:mod:`matlib.broadcasting`, :mod:`matlib.ops`),
linalg (:mod:`matlib.linalg`), shape manipulation
(:mod:`matlib.manipulation`), reductions (:mod:`matlib.stats`) and RNG
sampling (:mod:`matlib.rng`).

Note: re-exported names such as ``abs``, ``sum``, ``min`` and ``max``
intentionally shadow their builtins within this module.
"""

from .broadcasting import broadcast, elementwise
from .linalg import (
    cholesky,
    cross,
    det,
    inverse,
    matmul,
    matvec,
    outer,
    permute,
    solve,
    tensordot,
    transpose,
)
from .manipulation import concatenate, flip, split, stack, unbind
from .ops import abs, acos, asin, atan, cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh
from .rng import normal
from .stats import (
    max,
    max_axis,
    mean,
    mean_axis,
    std,
    sum,
    sum_axis,
    variance,
    variance_axis,
)

__all__ = [
    "abs",
    "acos",
    "asin",
    "atan",
    # broadcasting
    "broadcast",
    "cholesky",
    # manipulation
    "concatenate",
    "cos",
    "cosh",
    "cross",
    # linalg
    "det",
    "elementwise",
    # element-wise unary
    "exp",
    "flip",
    "inverse",
    "log",
    "matmul",
    "matvec",
    "max",
    "max_axis",
    "mean",
    "mean_axis",
    # rng
    "normal",
    "outer",
    "permute",
    "sin",
    "sinh",
    "solve",
    "split",
    "sqrt",
    "stack",
    "std",
    # stats – global
    "sum",
    # stats – axis-aware
    "sum_axis",
    "tan",
    "tanh",
    "tensordot",
    "transpose",
    "unbind",
    "variance",
    "variance_axis",
]
