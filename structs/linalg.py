from .broadcasting import broadcast
from .utils import build_index, zeroes, set_nested

def transpose(array):
    if array.ndim != 2:
        raise ValueError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0])

    return [
        [array[r][c] for r in range(rows)]
        for c in range(cols)
    ]

def tensordot(a, b, axes=2):
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError("axes must be >= 0")

        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
        axes_a = list(axes_a)
        axes_b = list(axes_b)

    axes_a = [ax % a.ndim for ax in axes_a]
    axes_b = [ax % b.ndim for ax in axes_b]

    if len(axes_a) != len(axes_b):
        raise ValueError("Axes lengths differ.")

    for aa, bb in zip(axes_a, axes_b):
        if a.shape[aa] != b.shape[bb]:
            raise ValueError(
                f"Shape mismatch: {a.shape[aa]} != {b.shape[bb]}"
            )

    free_a = [i for i in range(a.ndim) if i not in axes_a]
    free_b = [i for i in range(b.ndim) if i not in axes_b]

    out_shape = (
        tuple(a.shape[i] for i in free_a) +
        tuple(b.shape[i] for i in free_b)
    )

    contract_shape = tuple(a.shape[i] for i in axes_a)

    result = zeroes(out_shape)

    for out_idx in a.indices(out_shape):
        a_free = out_idx[:len(free_a)]
        b_free = out_idx[len(free_a):]

        total = 0

        for contract in a.indices(contract_shape):
            a_idx = build_index(
                a.ndim,
                (free_a, a_free),
                (axes_a, contract),
            )

            b_idx = build_index(
                b.ndim,
                (free_b, b_free),
                (axes_b, contract),
            )

            total += a[a_idx] * b[b_idx]

        set_nested(result, out_idx, total)

    return result

def matvec(a, b):
    if a.shape[1] != b.shape[1]: raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned.")
    
    result = []
    for row in a:
        result.append(sum(r * v for r, v in zip(row, b)))
    return result

def matmul(a, b):
    a_data, b_data, squeeze_type = broadcast(a, b)
    
    a_rows, a_cols = len(a_data), len(a_data[0]) if len(a_data) > 0 else 0
    b_rows, b_cols_count = len(b_data), len(b_data[0]) if len(b_data) > 0 else 0

    if a_cols != b_rows:
        raise ValueError(f"Matrix shapes not aligned: {a_cols} != {b_rows}")

    b_cols = list(zip(*b_data))
    result = []
    for row in a_data:
        new_row = []
        for col in b_cols:
            new_row.append(sum(r * c for r, c in zip(row, col)))
        result.append(new_row)

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
