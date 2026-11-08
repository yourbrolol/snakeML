"""Public math functions routed to an implementation by operand type."""

from __future__ import annotations

from typing import Any

from .router import route


@route
def broadcast(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.broadcast(*args, **kwargs)


@route
def elementwise(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.elementwise(*args, **kwargs)


@route
def cholesky(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.cholesky(*args, **kwargs)


@route
def cross(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.cross(*args, **kwargs)


@route
def det(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.det(*args, **kwargs)


@route
def inverse(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.inverse(*args, **kwargs)


@route
def matmul(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.matmul(*args, **kwargs)


@route
def matvec(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.matvec(*args, **kwargs)


@route
def outer(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.outer(*args, **kwargs)


@route
def permute(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.permute(*args, **kwargs)


@route
def solve(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.solve(*args, **kwargs)


@route
def tensordot(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.tensordot(*args, **kwargs)


@route
def transpose(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.transpose(*args, **kwargs)


@route
def concatenate(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.concatenate(*args, **kwargs)


@route
def flip(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.flip(*args, **kwargs)


@route
def split(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.split(*args, **kwargs)


@route
def stack(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.stack(*args, **kwargs)


@route
def unbind(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.unbind(*args, **kwargs)


@route
def abs(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.abs(*args, **kwargs)


@route
def acos(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.acos(*args, **kwargs)


@route
def asin(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.asin(*args, **kwargs)


@route
def atan(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.atan(*args, **kwargs)


@route
def cos(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.cos(*args, **kwargs)


@route
def cosh(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.cosh(*args, **kwargs)


@route
def exp(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.exp(*args, **kwargs)


@route
def log(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.log(*args, **kwargs)


@route
def sin(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.sin(*args, **kwargs)


@route
def sinh(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.sinh(*args, **kwargs)


@route
def sqrt(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.sqrt(*args, **kwargs)


@route
def tan(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.tan(*args, **kwargs)


@route
def tanh(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.tanh(*args, **kwargs)


@route(default="_gen")
def normal(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.normal(*args, **kwargs)


@route
def max(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.max(*args, **kwargs)


@route
def max_axis(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.max_axis(*args, **kwargs)


@route
def mean(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.mean(*args, **kwargs)


@route
def mean_axis(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.mean_axis(*args, **kwargs)


@route
def std(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.std(*args, **kwargs)


@route
def sum(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.sum(*args, **kwargs)


@route
def sum_axis(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.sum_axis(*args, **kwargs)


@route
def variance(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.variance(*args, **kwargs)


@route
def variance_axis(module: Any, *args: Any, **kwargs: Any) -> Any:
    return module.variance_axis(*args, **kwargs)


__all__ = [
    "abs",
    "acos",
    "asin",
    "atan",
    "broadcast",
    "cholesky",
    "concatenate",
    "cos",
    "cosh",
    "cross",
    "det",
    "elementwise",
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
    "sum",
    "sum_axis",
    "tan",
    "tanh",
    "tensordot",
    "transpose",
    "unbind",
    "variance",
    "variance_axis",
]
