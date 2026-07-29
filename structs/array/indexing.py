from debug import get_logger
from debug.errors import TypeMismatchError

logger = get_logger(__name__)


def _normalize_key(ndim, _key):
    """Normalize and expand indexing keys including slices and Ellipsis to match array rank."""
    key = _key if isinstance(_key, tuple) else (_key,)

    if key.count(Ellipsis) > 1:
        logger.error("invalid indexing key contains multiple ellipses", key=key)
        raise IndexError("__getitem__ should at most contain one Ellipsis (...)")

    cleaned = []
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


def _coerce_value(value):
    """Coerce Array instances or nested sequences into plain Python lists for item assignment."""
    from structs.array.array import Array

    if isinstance(value, Array):
        return value.data
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    return value


def _getitem(array, key, plain=False):
    key = _normalize_key(array.ndim, key)
    offset, shape, strides = array.offset, [], []
    ax = 0
    for idx in key:
        stride = array.strides[ax]
        size = array.shape[ax]
        if isinstance(idx, int):
            if idx < 0: idx += size
            offset += idx * stride
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(size)
            offset += start * stride
            shape.append(((stop - start + step - 1) // step))
            strides.append(stride * step)
        ax += 1
    if not plain:
        return [
            array.data,
            shape,
            strides,
            offset
        ]
    else: raise NotImplementedError

def __getitem(array, _key, target=None, depth=0):
    """Recursively retrieve items or slices from nested array list data."""
    if depth == 0:
        key = _normalize_key(array.ndim, _key)
    else:
        key = _key

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


def _setitem(array, _key, value, target=None, depth=0):
    """Recursively set values or slices within nested array list data."""
    if depth == 0:
        key = _normalize_key(array.ndim, _key)
    else:
        key = _key

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
            logger.error("unsupported index type in list/tuple", index_type=type(head).__name__)
            raise TypeMismatchError(f"Unsupported index type in list/tuple: {type(head).__name__}")
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
