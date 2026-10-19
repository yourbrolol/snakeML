"""Broadcasting and element-wise operators for nested-list data.

Provides the shape-normalization helpers used by
:class:`structs.array.array.Array` before dispatching binary operations,
plus the recursive scalar broadcast engine itself.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

__all__ = ["broadcast", "elementwise"]


def broadcast(a: Any, b: Any) -> Tuple[Any, Any, str]:
    """Normalize input array structures for compatible broadcasting and matrix operations.

    1-D inputs are reshaped towards alignment with their counterpart
    (``a`` becomes a row, ``b`` becomes a column vector); scalar inputs
    are wrapped into single-element containers. The returned
    ``squeeze_type`` records how to project the operation result back.

    Args:
        a: Left operand: :class:`~structs.array.array.Array`, list or scalar.
        b: Right operand in the same accepted forms.

    Returns:
        ``(a_data, b_data, squeeze_type)`` where ``a_data``/``b_data`` are
        normalized nested lists (via ``.data`` when the input exposes it)
        and ``squeeze_type`` is one of ``"none"``, ``"row"``,
        ``"column"`` or ``"scalar"``.
    """

    def _is_1d(obj: Any) -> bool:
        """Return whether *obj* behaves like a one-dimensional sequence."""
        if hasattr(obj, "ndim"):
            try:
                return obj.ndim == 1
            except Exception:
                pass
        if isinstance(obj, list):
            # empty list: treat as 1d
            if len(obj) == 0:
                return True
            # 1d if elements are not lists
            return not isinstance(obj[0], list)
        return False

    a_is_1d = _is_1d(a)
    b_is_1d = _is_1d(b)

    a_data = a.data if hasattr(a, "data") else a
    b_data = b.data if hasattr(b, "data") else b

    # Normalize to nested-list shapes: (N,) -> (1, N) and (N,) -> (N, 1)
    if a_is_1d:
        if isinstance(a_data, list):
            a_data = [a_data]
        else:
            a_data = [[a_data]]
    if b_is_1d:
        # ensure b_data is iterable
        if isinstance(b_data, list):
            b_data = [[x] for x in b_data]
        else:
            b_data = [[b_data]]

    if a_is_1d and b_is_1d:
        squeeze_type = "scalar"
    elif b_is_1d:
        squeeze_type = "column"
    elif a_is_1d:
        squeeze_type = "row"
    else:
        squeeze_type = "none"

    return a_data, b_data, squeeze_type


def elementwise(a: Any, b: Any, op: Callable[[Any, Any], Any]) -> Any:
    """Applies a math operator element-wise between self and another Array/scalar.

    Lists of equal length pair positionally; a length-1 list broadcasts
    across its counterpart; a single list operand broadcasts its scalar
    partner. Mismatched lengths raise :class:`ValueError`.

    Args:
        a: Left operand (nested list, scalar or array-like).
        b: Right operand in the same accepted forms.
        op: Two-argument callable applied to each broadcast pair.

    Returns:
        The nested-list result of broadcasting ``op`` over the pair.

    Raises:
        ValueError: If two list operands have unequal, non-broadcastable
            lengths.
    """
    # both scalars
    if not isinstance(a, list) and not isinstance(b, list):
        return op(a, b)

    # both lists: recurse element-wise; require same shape or zip will truncate
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            return [elementwise(x, y, op) for x, y in zip(a, b)]

        if len(a) == 1:
            return [elementwise(a[0], y, op) for y in b]

        if len(b) == 1:
            return [elementwise(x, b[0], op) for x in a]

        raise ValueError("Shapes cannot be broadcast")

    # one side is list: broadcast scalar across list
    if isinstance(a, list):
        return [elementwise(item_a, b, op) for item_a in a]
    if isinstance(b, list):
        return [elementwise(a, item_b, op) for item_b in b]
