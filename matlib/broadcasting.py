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

def _get_shape(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])


def _broadcast_shapes(shape_a, shape_b):
    ndim = max(len(shape_a), len(shape_b))
    shape_a = (1,) * (ndim - len(shape_a)) + shape_a
    shape_b = (1,) * (ndim - len(shape_b)) + shape_b
    out = []
    for dim_a, dim_b in zip(shape_a, shape_b):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            out.append(max(dim_a, dim_b))
        else:
            raise ValueError("Shapes cannot be broadcast")
    return tuple(out)


def _elementwise_broadcast(a, b, op, shape_a, shape_b, out_shape, ndim_a=None, ndim_b=None):
    if not out_shape:
        return op(a, b)

    if ndim_a is None:
        ndim_a = len(shape_a)
    if ndim_b is None:
        ndim_b = len(shape_b)

    ndim = len(out_shape)
    shape_a = (1,) * (ndim - len(shape_a)) + shape_a
    shape_b = (1,) * (ndim - len(shape_b)) + shape_b

    padded_a = ndim_a < ndim
    padded_b = ndim_b < ndim

    out_len = out_shape[0]
    result = []
    for i in range(out_len):
        if padded_a:
            elem_a = a
        elif shape_a[0] == 1:
            elem_a = a[0]
        else:
            elem_a = a[i]

        if padded_b:
            elem_b = b
        elif shape_b[0] == 1:
            elem_b = b[0]
        else:
            elem_b = b[i]

        next_ndim_a = ndim_a if padded_a else ndim_a - 1
        next_ndim_b = ndim_b if padded_b else ndim_b - 1

        result.append(
            _elementwise_broadcast(
                elem_a,
                elem_b,
                op,
                _get_shape(elem_a) if isinstance(elem_a, list) else (),
                _get_shape(elem_b) if isinstance(elem_b, list) else (),
                out_shape[1:],
                next_ndim_a,
                next_ndim_b,
            )
        )
    return result


def elementwise(a, b, op):
    """Applies a math operator element-wise between self and another Array/scalar."""
    if not isinstance(a, list) and not isinstance(b, list):
        return op(a, b)

    if isinstance(a, list) and isinstance(b, list):
        shape_a = _get_shape(a)
        shape_b = _get_shape(b)
        out_shape = _broadcast_shapes(shape_a, shape_b)
        return _elementwise_broadcast(a, b, op, shape_a, shape_b, out_shape)

    if isinstance(a, list):
        return [elementwise(item_a, b, op) for item_a in a]
    if isinstance(b, list):
        return [elementwise(a, item_b, op) for item_b in b]
