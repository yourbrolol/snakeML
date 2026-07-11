def _normalize_key(ndim, _key):
    key = _key if isinstance(_key, tuple) else (_key,)
    
    if key.count(Ellipsis) > 1:
        raise IndexError("__getitem__ should at most contain one Ellipsis (...)!")
        
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
        return _getitem(array, tail, target[head], depth+1)

    if isinstance(head, slice):
        return [_getitem(array, tail, item, depth+1) for item in target[head]]

    raise TypeError(f"Unsupported index type: {type(head).__name__}")

def _setitem(array, _key, value, target=None, depth=0):
    key = _key
    if depth == 0: key = _normalize_key(array.ndim, _key)
    target = array.data if target is None else target
    if len(key) == 1:
        k = key[0]
        if isinstance(k, slice):
            target[k] = value if isinstance(value, list) else [value]
        else:
            target[k] = value
        return

    head, tail = key[0], key[1:]

    if isinstance(head, slice):
        children = target[head]
        for i, child in enumerate(children):
            sub_val = value[i] if isinstance(value, list) and i < len(value) else value
            _setitem(array, tail, sub_val, child, depth+1)
    else:
        _setitem(array, tail, value, target[head], depth+1)


