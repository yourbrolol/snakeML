"""Math kernel library for snakeML.

Public functions are routed to an implementation for the operand type.
Nested lists use :mod:`matlib._list`, Array instances use
:mod:`matlib._array`, and representation-independent helpers live in
:mod:`matlib._gen`.

Note: re-exported names such as ``abs``, ``sum``, ``min`` and ``max``
intentionally shadow their builtins within this module.
"""

from .protocol import (
    abs,
    acos,
    asin,
    atan,
    broadcast,
    cholesky,
    concatenate,
    cos,
    cosh,
    cross,
    det,
    elementwise,
    exp,
    flip,
    inverse,
    log,
    matmul,
    matvec,
    max,
    max_axis,
    mean,
    mean_axis,
    normal,
    outer,
    permute,
    sin,
    sinh,
    solve,
    split,
    sqrt,
    stack,
    std,
    sum,
    sum_axis,
    tan,
    tanh,
    tensordot,
    transpose,
    unbind,
    variance,
    variance_axis,
)
from .protocol import __all__

__all__ = list(__all__)
