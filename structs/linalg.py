from .broadcasting import broadcast

def transpose(array):
    if array.ndim != 2:
        raise ValueError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0])

    return [
        [array[r][c] for r in range(rows)]
        for c in range(cols)
    ]

def matvec(a, b):
    if a.shape[1] != b.shape[1]: raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned.")
    
    result = []
    for row in a:
        result.append(sum(r * v for r, v in zip(row, b)))
    return result

def matmul(a, b):
    # Prepare data and shapes via broadcasting function
    a_data, b_data, squeeze_type = broadcast(a, b)
    
    # Simple, strict 2D shape alignment validation
    a_rows, a_cols = len(a_data), len(a_data[0]) if len(a_data) > 0 else 0
    b_rows, b_cols_count = len(b_data), len(b_data[0]) if len(b_data) > 0 else 0

    if a_cols != b_rows:
        raise ValueError(f"Matrix shapes not aligned: {a_cols} != {b_rows}")

    # Core 2D matrix multiplication kernel
    b_cols = list(zip(*b_data))
    result = []
    for row in a_data:
        new_row = []
        for col in b_cols:
            new_row.append(sum(r * c for r, c in zip(row, col)))
        result.append(new_row)

    # Post-processing: Squeeze output back based on the broadcast instructions
    if squeeze_type == 'scalar':
        return result[0][0]
    if squeeze_type == 'column':
        return [row[0] for row in result]
    if squeeze_type == 'row':
        return result[0]
        
    return result

def outer(a, b):
    if a.ndim != b.ndim: raise NotImplementedError("Outer product only implemented for same length Tensors.")
    return [[i*j for j in b] for i in a]
