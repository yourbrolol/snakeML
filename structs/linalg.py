def transpose(array):
    if array.ndim != 2:
        raise ValueError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0])

    return [
        [array[r][c] for r in range(rows)]
        for c in range(cols)
    ]

def outer(a, b):
    if a.ndim != b.ndim: raise NotImplementedError("Outer product only implemented for same length Tensors.")
    return [[i*j for j in b] for i in a]
