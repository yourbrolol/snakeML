"""Shared type aliases used across :mod:`structs`.

Every public function in :mod:`structs` is annotated with the aliases defined
here, so that reading a signature tells you exactly which container types,
scalar types and callables a function accepts.

The aliases are plain assignments (not ``TypeAlias``) so the module stays
importable on Python 3.9 as declared in ``pyproject.toml``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, List, Sequence, Tuple, TypeVar, Union

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from structs.array.array import Array

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:  # pragma: no cover - Python 3.9 fallback
    EllipsisType = Any

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover - Python 3.9/3.10 fallback
    Self = Any


# --------------------------------------------------------------------------- #
# Scalars and element types
# --------------------------------------------------------------------------- #

#: A single numeric/logical element stored inside an ``Array``.
Scalar = Union[bool, int, float, complex]

#: A nested, row-major container mirroring the shape of an array.
NestedList = Union[Scalar, List["NestedList"], Tuple["NestedList", ...]]

#: Any value that can be nested into an :class:`~structs.array.array.Array`.
ArrayLike = Union[NestedList, range, Any]

#: The two-argument callable applied to each broadcast pair (e.g. ``x + y``).
BinaryOp = Callable[[Any, Any], Any]

#: The single-argument callable applied to each element (e.g. ``math.sqrt``).
UnaryOp = Callable[[Any], Any]


# --------------------------------------------------------------------------- #
# Shapes and axes
# --------------------------------------------------------------------------- #

#: A single axis index. Negative values count from the end.
Axis = int

#: An ordered collection of axis indices, e.g. ``(0, 2)``.
Axes = Sequence[int]

#: Either one axis or several at once, e.g. ``0`` or ``(0, 2)``.
AxisArg = Union[Axis, Axes]

#: A user-supplied shape: a bare ``int`` or a sequence of dimensions.
ShapeArg = Union[int, Sequence[int]]

#: A fully resolved shape as stored on an array: one ``int`` per dimension.
Shape = Tuple[int, ...]

#: Resolved strides (element offsets per dimension), same layout as ``Shape``.
Strides = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #

#: A fancy-index payload: integer positions or boolean masks.
IndexList = Union[Sequence[int], Sequence[bool]]

#: Everything allowed in a single position of an index key.
IndexElement = Union[int, slice, bool, IndexList, EllipsisType]

#: A full index key: one element, or a tuple covering some/all dimensions.
IndexKey = Union[IndexElement, Tuple[IndexElement, ...]]


# --------------------------------------------------------------------------- #
# Construction and conversion
# --------------------------------------------------------------------------- #

#: A literal value (or ``None``), or a zero-argument callable producing one
#: per element. ``None`` is legal because ``Array.empty`` fills with it.
FillValue = Union[Scalar, None, Callable[[], Scalar]]

#: dtype labels produced by :attr:`Array.dtype` / accepted by ``astype``.
DTypeName = str

#: Which side to insert on for ``searchsorted``.
SearchSide = str

#: Padding strategy accepted by :meth:`Array.pad`.
PadMode = str


# --------------------------------------------------------------------------- #
# Type variables
# --------------------------------------------------------------------------- #

#: Return type for methods that construct ``self.__class__`` so that
#: subclasses keep their own type through the fluent API.
ArrayT = TypeVar("ArrayT", bound="Array")

#: Return type for instance methods that build a new array from ``self``,
#: mirroring ``typing.Self`` so subclasses keep their concrete type.
ReturnSelf = Self