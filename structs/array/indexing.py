"""Private recursive indexing engine backing ``Array.__getitem__``/``__setitem__``.

Keys are first normalised to exactly ``ndim`` positional entries, then applied
one dimension at a time to the nested list representation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

from debug import get_logger
from debug.errors import TypeMismatchError
from structs.dtypes import IndexElement, IndexKey

logger = get_logger(__name__)

__all__ = [
    "_normalize_key",
    "_coerce_value",
    "_getitem",
    "_setitem",
]

#: A key guaranteed to be a tuple with exactly one entry per dimension.
NormalizedKey = Tuple[IndexElement, ...]


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
        IndexError: If the key contains more than one ``Ellipsis``.

    Example:
        >>> _normalize_key(3, (0, Ellipsis))
        (0, slice(None, None, None), slice(None, None, None))
    """
    key = _key if isinstance(_key, tuple) else (_key,)

    if key.count(Ellipsis) > 1:
        logger.error("invalid indexing key contains multiple ellipses", key=key)
        raise IndexError("__getitem__ should at most contain one Ellipsis (...)")

    cleaned: List[IndexElement] = []
    for idx in key:
        if idx is Ellipsis:
            pad_count = ndim - (len(key) - 1)
            cleaned.extend([slice(None)] * pad_count)
        else:
            cleaned.append(idx)

    if len(cleaned) < ndim:
        cleaned.extend([slice(None)] * (ndim - len(cleaned)))

    logger.debug("normalized key", cleaned=cleaned)

    return tuple(cleaned)


def _coerce_value(value: Any) -> Any:
    """Coerce Array instances or nested sequences into plain Python lists for item assignment.

    Args:
        value: A scalar, ``Array``, ``list``, or ``tuple``.

    Returns:
        Any: ``Array`` inputs are replaced by their underlying ``data``;
        ``list``/``tuple`` inputs are rebuilt recursively as lists; scalars are
        returned unchanged.
    """
    from structs.array.array import Array

    if isinstance(value, Array):
        return value.data
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    return value


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
        Any: A scalar for a single int/slice-with-no-tail selection, or a
        nested ``list`` mirroring the selected sub-shape.

    Raises:
        IndexError: If a boolean mask length does not match the dimension.
        TypeMismatchError: If the leading index entry is not an int, slice,
            boolean mask or integer list.

    Example:
        >>> Array([[1, 2], [3, 4]])[0]
        [1, 2]
    """
    key: NormalizedKey = (
        _normalize_key(array.ndim, _key)
        if depth == 0
        else cast("NormalizedKey", _key)
    )

    target = array.data if target is None else target

    if not key:
        return target

    head, *tail = key

    if isinstance(head, (list, tuple)):
        if all(isinstance(item, bool) for item in head):
            if len(head) != len(target):
                raise IndexError("boolean index length mismatch")
            selected = [item for item, keep in zip(target, head) if keep]
            if not tail:
                return selected
            return [_getitem(array, tuple(tail), item, depth + 1) for item in selected]

        if all(isinstance(item, int) for item in head):
            selected = [target[item] for item in head]
            if not tail:
                return selected
            return [_getitem(array, tuple(tail), item, depth + 1) for item in selected]

    if isinstance(head, int):
        if not tail:
            return target[head]
        return _getitem(array, tuple(tail), target[head], depth + 1)

    if isinstance(head, slice):
        return [_getitem(array, tuple(tail), item, depth + 1) for item in target[head]]

    logger.error("unsupported index type", index_type=type(head).__name__)
    raise TypeMismatchError(f"Unsupported index type: {type(head).__name__}")


def _setitem(
    array: Any,
    _key: IndexKey,
    value: Any,
    target: Optional[Any] = None,
    depth: int = 0,
) -> None:
    """Recursively set values or slices within nested array list data.

    Assignments mutate ``array.data`` in place; no new array is produced.

    Args:
        array: The target ``Array``; provides ``ndim`` and the default target.
        _key: The index key for this level. Normalised only at ``depth == 0``.
        value: Scalar or nested sequence to assign. ``Array``/tuple inputs are
            coerced to lists first, and scalars are broadcast across a slice.
        target: Sub-container being written to. Defaults to ``array.data``; the
            recursion passes each selected child down.
        depth: Current recursion depth, used to normalise the key exactly once.

    Raises:
        IndexError: If a boolean mask length does not match the dimension.
        TypeMismatchError: If the leading index entry of a list/tuple key holds
            values that are neither all bools nor all ints.

    Example:
        >>> a = Array([[1, 2], [3, 4]])
        >>> a[0, 1] = 9
        >>> a
        [[1, 9], [3, 4]]
    """
    key: NormalizedKey = (
        _normalize_key(array.ndim, _key)
        if depth == 0
        else cast("NormalizedKey", _key)
    )

    target = array.data if target is None else target
    value = _coerce_value(value)

    if len(key) == 1:
        k = key[0]
        if isinstance(k, slice):
            if isinstance(value, list):
                target[k] = value
            else:
                target[k] = [value] * len(range(*k.indices(len(target))))
        else:
            target[k] = value
        return

    head, tail = key[0], key[1:]

    if isinstance(head, (list, tuple)):
        if all(isinstance(item, bool) for item in head):
            if len(head) != len(target):
                raise IndexError("boolean index length mismatch")
            selected_indices = [i for i, keep in enumerate(head) if keep]
            if isinstance(value, list):
                for idx, val in zip(selected_indices, value):
                    _setitem(array, tail, val, target[idx], depth + 1)
            else:
                for idx in selected_indices:
                    _setitem(array, tail, value, target[idx], depth + 1)
        elif all(isinstance(item, int) for item in head):
            if isinstance(value, list):
                for idx, val in zip(head, value):
                    _setitem(array, tail, val, target[idx], depth + 1)
            else:
                for idx in head:
                    _setitem(array, tail, value, target[idx], depth + 1)
        else:
            logger.error(
                "unsupported index type in list/tuple", index_type=type(head).__name__
            )
            raise TypeMismatchError(
                f"Unsupported index type in list/tuple: {type(head).__name__}"
            )
    elif isinstance(head, slice):
        children = target[head]
        if isinstance(value, list):
            for i, child in enumerate(children):
                sub_val = value[i] if i < len(value) else value
                _setitem(array, tail, sub_val, child, depth + 1)
        else:
            for child in children:
                _setitem(array, tail, value, child, depth + 1)
    else:
        _setitem(array, tail, value, target[head], depth + 1)