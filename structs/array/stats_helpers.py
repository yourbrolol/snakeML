"""Statistic and reduction helpers for the Array bufferization roadmap.

Exports the same names as ``structs/array/utils.py`` for backward compatibility:
``fill``, ``where``, ``unique``, ``sort``, ``argsort``, ``astype``, plus
any others that will be added during the Milestones 4‑6 refactor.

Each name is imported from ``structs/array/utils.py`` so that existing imports
(e.g. ``from structs.array.utils import fill``) continue to work unchanged.
"""

from __future__ import annotations

from structs.array.utils import (
    fill,
    where,
    unique,
    sort,
    argsort,
    astype,
)

__all__ = [
    "fill",
    "where",
    "unique",
    "sort",
    "argsort",
    "astype",
]