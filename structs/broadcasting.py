def broadcast(a, b):
    a_is_1d = getattr(a, 'ndim', 1) == 1 or (isinstance(a, list) and not isinstance(a[0], list))
    b_is_1d = getattr(b, 'ndim', 1) == 1 or (isinstance(b, list) and not isinstance(b[0], list))

    a_data = a.data if hasattr(a, 'data') else a
    b_data = b.data if hasattr(b, 'data') else b

    if a_is_1d:
        a_data = [a_data]       # (N,) -> (1, N)
    if b_is_1d:
        b_data = [[x] for x in b_data]  # (N,) -> (N, 1)

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
    if not isinstance(a, list) and not isinstance(b, list):
        return op(a, b)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Operands could not be broadcast together due to shape mismatch.")
        return [elementwise(item_a, item_b, op) for item_a, item_b in zip(a, b)]
    # Broadcasting scalar
    if isinstance(a, list):
        return [elementwise(item_a, b, op) for item_a in a]
    if isinstance(b, list):
        return [elementwise(a, item_b, op) for item_b in b]
