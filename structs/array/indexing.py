from debug import get_logger
from debug.errors import TypeMismatchError

logger = get_logger(__name__)


def _normalize_key(ndim, _key):
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

    return tuple(cleaned)


def _coerce_value(value):
    from structs.array.array import Array

    if isinstance(value, Array):
        return value.data
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    return value


def _getitem(array, _key, target=None, depth=0):
    if depth == 0:
        key = _normalize_key(array.ndim, _key)
    else:
        key = _key

    target = array.data if target is None else target

    if not key:
        return target

    head, *tail = key

    if isinstance(head, int):
        if not tail:
            return target[head]
        return _getitem(array, tuple(tail), target[head], depth + 1)

    if isinstance(head, slice):
        return [_getitem(array, tuple(tail), item, depth + 1) for item in target[head]]

    logger.error("unsupported index type", index_type=type(head).__name__)
    raise TypeMismatchError(f"Unsupported index type: {type(head).__name__}")


def _setitem(array, _key, value, target=None, depth=0):
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

    if isinstance(head, slice):
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


