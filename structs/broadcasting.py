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

