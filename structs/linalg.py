def transpose(array):
    if array.ndim != 2:
        raise ValueError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0])

    return [
        [array[r][c] for r in range(rows)]
        for c in range(cols)
    ]

def matmul(a, b):
    if (a.ndim or b.ndim) != 2: raise NotImplementedError(f"A: {a.ndim} or B: {b.ndim} is not a matrice!")
    if a.shape[1] != b.shape[0]: raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned.")

    b_cols = list(zip(*b.data))
    
    result = []
    for row in a:
        new_row = []
        for col in b_cols:
            new_row.append(sum(r * c for r, c in zip(row, col)))
        result.append(new_row)
    return result

def outer(a, b):
    if a.ndim != b.ndim: raise NotImplementedError("Outer product only implemented for same length Tensors.")
    return [[i*j for j in b] for i in a]
