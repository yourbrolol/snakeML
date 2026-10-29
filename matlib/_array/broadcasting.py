"""Temporary Array bridge for nested-list broadcasting kernels."""

from .._list.broadcasting import *
from .._list.broadcasting import __all__


def elementwise(a, b, op):
    # for i, j in zip(a.data.buffer, b.data.buffer): op(i, j)
    pass
