def broadcast(a, b):
    """Normalize input array structures for compatible broadcasting and matrix operations."""
    def _is_1d(obj):
        if hasattr(obj, 'ndim'):
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

    a_data = a.data if hasattr(a, 'data') else a
    b_data = b.data if hasattr(b, 'data') else b

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
        squeeze_type = 'scalar'
    elif b_is_1d:
        squeeze_type = 'column'
    elif a_is_1d:
        squeeze_type = 'row'
    else:
        squeeze_type = 'none'

    return a_data, b_data, squeeze_type

def elementwise(a, b, op):
    """Applies a math operator element-wise between self and another Array/scalar."""
    # both scalars
    if not isinstance(a, list) and not isinstance(b, list):
        return op(a, b)

    # both lists: recurse element-wise; require same shape or zip will truncate
    if isinstance(a, list) and isinstance(b, list):
        return [elementwise(item_a, item_b, op) for item_a, item_b in zip(a, b)]

    # one side is list: broadcast scalar across list
    if isinstance(a, list):
        return [elementwise(item_a, b, op) for item_a in a]
    if isinstance(b, list):
        return [elementwise(a, item_b, op) for item_b in b]
