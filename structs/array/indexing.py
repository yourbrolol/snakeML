"""Private recursive indexing engine backing ``Array.__getitem__``/``__setitem__``.

Keys are first normalised to exactly ``ndim`` positional entries, then applied
one dimension at a time to the nested list or flat ``Buffer`` representation.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple, cast
from collections import deque

from debug import get_logger
from debug.errors import TypeMismatchError
from structs.dtypes import IndexElement, IndexKey
from structs.array.buffer import Buffer

logger = get_logger(__name__)

__all__ = [
    "_normalize_key",
    "_coerce_value",
    "_getitem",
    "_setitem",
]

#: A key guaranteed to be a tuple with exactly one entry per dimension.
NormalizedKey = Tuple[IndexElement, ...]


def _is_int_index(value: Any) -> bool:
    """Return True for genuine integer indices (``bool`` is not an int here)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_int(index: int, dim: int) -> int:
    """Resolve a possibly negative position and validate it against ``dim``.

    Args:
        index: The user-supplied integer position.
        dim: Size of the axis being indexed.

    Returns:
        int: A non-negative position in ``[0, dim)``.

    Raises:
        IndexError: If the position is out of bounds.
    """
    resolved = index + dim if index < 0 else index
    if resolved < 0 or resolved >= dim:
        raise IndexError(f"index {index} is out of bounds for axis with size {dim}")
    return resolved


def _indices_for(head: Any, dim: int) -> List[int]:
    """Expand a slice or fancy index into resolved integer positions.

    Args:
        head: A ``slice``, an integer list, or a boolean mask.
        dim: Size of the axis being indexed.

    Returns:
        List[int]: Positions selected along the axis, in order.

    Raises:
        IndexError: If a boolean mask length does not match ``dim``, or an
            integer position is out of bounds.
        TypeMismatchError: If the list contains neither all bools nor all ints.
    """
    if isinstance(head, slice):
        return list(range(*head.indices(dim)))

    if isinstance(head, (list, tuple)):
        if len(head) == 0:
            return []

        if all(isinstance(item, bool) for item in head):
            if len(head) != dim:
                raise IndexError(
                    f"boolean index length {len(head)} does not match axis size {dim}"
                )
            return [i for i, keep in enumerate(head) if keep]

        if all(_is_int_index(item) for item in head):
            return [_normalize_int(item, dim) for item in head]

        raise TypeMismatchError(
            "A list/tuple index must contain either all booleans or all integers"
        )

    raise TypeMismatchError(f"Unsupported index type: {type(head).__name__}")


def _selected_dim_count(key: NormalizedKey, start: int = 0) -> int:
    """Count axes in ``key[start:]`` that keep their dimension (non-int)."""
    return sum(1 for entry in key[start:] if not _is_int_index(entry))


def _value_ndim(value: Any) -> int:
    """Return how many nested list dimensions ``value`` has (0 for a scalar)."""
    if isinstance(value, (list, tuple)):
        if not value:
            return 1
        return 1 + _value_ndim(value[0])
    return 0


def _scalar_value(value: Any) -> Any:
    """Unwrap single-element sequences destined for a scalar position."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalar_value(value[0])
        raise IndexError("cannot assign a sequence to a scalar element")
    if isinstance(value, Buffer):
        if len(value) == 1:
            return value[0]
        raise IndexError("cannot assign a sequence to a scalar element")
    return value


def _broadcast_sequence(
    indices: List[int], value: Any, remaining_dims: int
) -> Iterator[Tuple[int, Any]]:
    """Pair selected positions with the slice of ``value`` that belongs to them.

    Value dimensions are aligned to the trailing selected dimensions, mirroring
    NumPy broadcasting. A scalar or a lower-rank value is broadcast across the
    current axis.

    Args:
        indices: Positions selected along the current axis.
        value: Value supplied for the selection.
        remaining_dims: Number of dimension-keeping axes from here onward.

    Yields:
        Tuple[int, Any]: ``(position, sub_value)`` pairs.

    Raises:
        IndexError: If ``value`` has too many dimensions or cannot be
            broadcast across the selection.
    """
    value_ndim = _value_ndim(value)

    if value_ndim == 0 or value_ndim < remaining_dims:
        for index in indices:
            yield index, value
        return

    if value_ndim > remaining_dims:
        raise IndexError(
            f"too many value dimensions: value has {value_ndim}, "
            f"selection expects {remaining_dims}"
        )

    if not isinstance(value, list):
        raise IndexError("expected a sequence for this dimension")

    count = len(value)
    if count == len(indices):
        for index, sub_value in zip(indices, value):
            yield index, sub_value
    elif count == 1 and len(indices) != 1:
        for index in indices:
            yield index, value[0]
    else:
        raise IndexError(
            f"could not broadcast value of length {count} "
            f"into selection of length {len(indices)}"
        )


def _normalize_key(ndim: int, _key: IndexKey) -> NormalizedKey:
    """Normalize and expand indexing keys including slices and Ellipsis to match array rank.

    Args:
        ndim: Rank of the array being indexed; the length of the result.
        _key: A bare index (wrapped into a 1-tuple) or an explicit tuple.

    Returns:
        NormalizedKey: A tuple of exactly ``ndim`` entries. A single
        ``Ellipsis`` is replaced by as many full ``slice(None)`` entries as the
        remaining dimensions require, and any missing trailing positions are
        padded the same way.

    Raises:
        IndexError: If the key contains more than one ``Ellipsis``, or more
            indices than the array has dimensions.

    Example:
        >>> _normalize_key(3, (0, Ellipsis))
        (0, slice(None, None, None), slice(None, None, None))
    """
    key = _key if isinstance(_key, tuple) else (_key,)

    ellipsis_positions = [i for i, entry in enumerate(key) if entry is Ellipsis]
    if len(ellipsis_positions) > 1:
        logger.error("invalid indexing key contains multiple ellipses", key=key)
        raise IndexError("An index can only contain a single Ellipsis (...)")

    if ellipsis_positions:
        position = ellipsis_positions[0]
        pad_count = ndim - (len(key) - 1)
        if pad_count < 0:
            raise IndexError(
                f"too many indices for array: array is {ndim}-dimensional, "
                f"but {len(key) - 1} were indexed"
            )
        cleaned: List[IndexElement] = (
            list(key[:position]) + [slice(None)] * pad_count + list(key[position + 1 :])
        )
    else:
        if len(key) > ndim:
            raise IndexError(
                f"too many indices for array: array is {ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
        cleaned = list(key) + [slice(None)] * (ndim - len(key))

    logger.debug("normalized key", cleaned=cleaned)

    return tuple(cleaned)


def _buffer_to_nested(
    buffer: Buffer,
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    offset: int,
    depth: int = 0,
) -> Any:
    """Rebuild a nested list from a flat buffer using shape/strides/offset."""
    if depth == len(shape):
        return buffer[offset]
    return [
        _buffer_to_nested(
            buffer, shape, strides, offset + i * strides[depth], depth + 1
        )
        for i in range(shape[depth])
    ]


def _coerce_value(value: Any) -> Any:
    """Coerce Array instances or nested sequences into plain Python lists for item assignment.

    Args:
        value: A scalar, ``Array``, ``Buffer``, ``list``, or ``tuple``.

    Returns:
        Any: ``Array`` inputs are replaced by their underlying nested data (a
        buffer-backed array is expanded through its shape/strides/offset);
        ``Buffer`` inputs become flat lists; ``list``/``tuple`` inputs are
        rebuilt recursively as lists; scalars are returned unchanged.
    """
    from structs.array.array import Array

    if isinstance(value, Array):
        if isinstance(value.data, Buffer):
            return _buffer_to_nested(
                value.data, value.shape, value.strides, value.offset
            )
        return value.data
    if isinstance(value, Buffer):
        return list(value.buffer)
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    return value


def _result_shape(key, shape):
    result = []

    for head, size in zip(key, shape):
        if isinstance(head, bool):
            raise TypeMismatchError("Boolean scalar indices are not supported")

        if _is_int_index(head):
            _normalize_int(cast(int, head), size)

        elif isinstance(head, slice):
            start, stop, step = head.indices(size)
            result.append(len(range(start, stop, step)))

        else:
            result.append(len(_indices_for(head, size)))

    return tuple(result)


def _getitem_buffer(
    buffer: Buffer,
    key: NormalizedKey,
    offset: int,
    shape: deque[int] | Tuple[int, ...],
    strides: deque[int] | Tuple[int, ...],
) -> Any:
    """Read from flat buffer storage using resolved shape/strides.

    Integer/slice-only indexing produces a lazy view.
    Fancy indexing materialises a copy.
    """
    from structs.array.array import Array

    shape_t = tuple(shape)
    strides_t = tuple(strides)

    cur_offset = offset
    out_shape: List[int] = []
    out_strides: List[int] = []

    for depth, head in enumerate(key):
        if isinstance(head, bool):
            raise TypeMismatchError("Boolean scalar indices are not supported")

        if _is_int_index(head):
            index = _normalize_int(cast(int, head), shape_t[depth])
            cur_offset += index * strides_t[depth]
            continue

        if isinstance(head, slice):
            start, stop, step = head.indices(shape_t[depth])
            count = len(range(start, stop, step))

            cur_offset += start * strides_t[depth]
            out_shape.append(count)
            out_strides.append(strides_t[depth] * step)
            continue

        # Fancy indexing: materialize from this axis onward.
        indices = _indices_for(head, shape_t[depth])

        remaining_key = key[depth + 1 :]
        remaining_shape = shape_t[depth + 1 :]
        remaining_strides = strides_t[depth + 1 :]

        # Dimensions preserved before the fancy index must
        # also be preserved in the resulting copy.
        prefix_shape = tuple(out_shape)

        nested = []
        for index in indices:
            sub = _getitem_buffer(
                buffer,
                remaining_key,
                cur_offset + index * strides_t[depth],
                remaining_shape,
                remaining_strides,
            )

            nested.append(sub.tolist() if isinstance(sub, Array) else sub)

        result_shape = (
            tuple(out_shape)
            + (len(indices),)
            + _result_shape(remaining_key, remaining_shape)
        )

        return Array(
            values=nested,
            shape=result_shape,
        )

    if not out_shape:
        return buffer[cur_offset]

    return Array(
        values=buffer,
        raw=True,
        shape=tuple(out_shape),
        strides=tuple(out_strides),
        offset=cur_offset,
    )


def _getitem(
    array: Any,
    _key: IndexKey,
    target: Optional[Any] = None,
    depth: int = 0,
) -> Any:
    """Recursively retrieve items or slices from nested array list data.

    Args:
        array: The source ``Array``; used for ``ndim`` and the default target.
        _key: The index key for this level. Normalised only at ``depth == 0``.
        target: Sub-container being indexed. Defaults to ``array.data``; the
            recursion passes each selected child down.
        depth: Current recursion depth, used to normalise the key exactly once.

    Returns:
        Any: A scalar for a pure int selection, or a nested ``list`` mirroring
        the selected sub-shape.

    Raises:
        IndexError: If a boolean mask length does not match the dimension, an
            integer position is out of bounds, or there are too many indices.
        TypeMismatchError: If the leading index entry is not an int, slice,
            boolean mask or integer list, or is a boolean scalar.

    Example:
        >>> Array([[1, 2], [3, 4]])[0]
        [1, 2]
    """
    key: NormalizedKey = (
        _normalize_key(array.ndim, _key) if depth == 0 else cast("NormalizedKey", _key)
    )

    target = array.data if target is None else target

    if isinstance(target, Buffer):
        return _getitem_buffer(
            target,
            key,
            array.offset if depth == 0 else 0,
            tuple(array.shape),
            tuple(array.strides),
        )

    if not key:
        return target


def _assign_buffer(
    buffer: Buffer,
    key: NormalizedKey,
    value: Any,
    offset: int,
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    depth: int = 0,
) -> None:
    """Assign ``value`` into flat buffer storage using shape/strides."""
    if depth == len(shape):
        buffer[offset] = _scalar_value(value)
        return

    head = key[depth]
    if isinstance(head, bool):
        raise TypeMismatchError("Boolean scalar indices are not supported")

    if _is_int_index(head):
        index = _normalize_int(cast(int, head), shape[depth])
        _assign_buffer(
            buffer,
            key,
            value,
            offset + index * strides[depth],
            shape,
            strides,
            depth + 1,
        )
        return

    indices = _indices_for(head, shape[depth])
    remaining = _selected_dim_count(key, depth)

    for index, sub_value in _broadcast_sequence(indices, value, remaining):
        _assign_buffer(
            buffer,
            key,
            sub_value,
            offset + index * strides[depth],
            shape,
            strides,
            depth + 1,
        )


def _setitem(
    array: Any,
    _key: IndexKey,
    value: Any,
    target: Optional[Any] = None,
    depth: int = 0,
) -> None:
    """Recursively set values or slices within nested array list data.

    Assignments mutate ``array.data`` in place; no new array is produced.
    Values broadcast following NumPy's trailing-dimension alignment.

    Args:
        array: The target ``Array``; provides ``ndim`` and the default target.
        _key: The index key for this level. Normalised only at ``depth == 0``.
        value: Scalar or nested sequence to assign. ``Array``/tuple inputs are
            coerced to lists first, and scalars are broadcast across a slice.
        target: Sub-container being written to. Defaults to ``array.data``; the
            recursion passes each selected child down.
        depth: Current recursion depth, used to normalise the key exactly once.

    Raises:
        IndexError: If a boolean mask length does not match the dimension, an
            integer position is out of bounds, assignment would resize the
            array, or the value cannot be broadcast to the selection.
        TypeMismatchError: If the leading index entry of a list/tuple key holds
            values that are neither all bools nor all ints.

    Example:
        >>> a = Array([[1, 2], [3, 4]])
        >>> a[0, 1] = 9
        >>> a
        [[1, 9], [3, 4]]
    """
    key: NormalizedKey = (
        _normalize_key(array.ndim, _key) if depth == 0 else cast("NormalizedKey", _key)
    )

    target = array.data if target is None else target
    if depth == 0:
        value = _coerce_value(value)

    if isinstance(target, Buffer):
        _assign_buffer(
            target,
            key,
            value,
            array.offset if depth == 0 else 0,
            array.shape,
            array.strides,
            depth,
        )
        return

    if not key:
        if isinstance(value, (list, tuple, Buffer)):
            raise IndexError("cannot assign a sequence to a 0-d array")
        array.data = value
        return

    if not isinstance(target, list):
        raise IndexError("too many indices for array")
