from .utils import build_index, zeroes, set_nested, indices


def transpose(array):
    if array.ndim != 2:
        raise ValueError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0]) if rows else 0

    return [[array[r][c] for r in range(rows)] for c in range(cols)]


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

#    for aa, bb in zip(axes_a, axes_b):
#        if a.shape[aa] != b.shape[bb]:
#            raise ValueError(f"Shape mismatch: {a.shape[aa]} != {b.shape[bb]}")

    free_a = [i for i in range(a.ndim) if i not in axes_a]
    free_b = [i for i in range(b.ndim) if i not in axes_b]

    out_shape = tuple(a.shape[i] for i in free_a) + tuple(b.shape[i] for i in free_b)
    contract_shape = tuple(a.shape[i] for i in axes_a)

    if not out_shape:
        total = 0
        for contract in indices(contract_shape):
            a_idx = build_index(a.ndim, (axes_a, contract))
            b_idx = build_index(b.ndim, (axes_b, contract))
            total += a[a_idx] * b[b_idx]
        return total

    result = zeroes(out_shape)

    for out_idx in indices(out_shape):
        a_free = out_idx[:len(free_a)]
        b_free = out_idx[len(free_a):]

        total = 0

        for contract in indices(contract_shape):
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
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned.")

    result = []
    for row in a:
        result.append(sum(r * v for r, v in zip(row, b)))
    return result


def matmul(a, b):
    a_data = a.data if hasattr(a, "data") else a
    b_data = b.data if hasattr(b, "data") else b

    if not isinstance(a_data, list):
        return a_data * b_data
    if not isinstance(b_data, list):
        return [x * b_data for x in a_data]
    if not isinstance(a_data[0], list):
        if not isinstance(b_data[0], list):
            return sum(x * y for x, y in zip(a_data, b_data))
        return [sum(x * col[j] for j, x in enumerate(a_data)) for col in zip(*b_data)]
    if not isinstance(b_data[0], list):
        return [sum(r * elem for r, elem in zip(row, b_data)) for row in a_data]

    if len(a_data[0]) != len(b_data):
        raise ValueError(f"Matrix shapes not aligned: {len(a_data[0])} != {len(b_data)}")

    return [
        [sum(row[k] * b_data[k][col_idx] for k in range(len(b_data))) for col_idx in range(len(b_data[0]))]
        for row in a_data
    ]


def outer(a, b):
    a_data = a.data if hasattr(a, "data") else a
    b_data = b.data if hasattr(b, "data") else b

    if not isinstance(a_data, list):
        a_data = [a_data]
    if not isinstance(b_data, list):
        b_data = [b_data]

    if isinstance(a_data[0], list):
        a_flat = [item for row in a_data for item in row]
    else:
        a_flat = a_data

    if isinstance(b_data[0], list):
        b_flat = [item for row in b_data for item in row]
    else:
        b_flat = b_data

    return [[a_val * b_val for b_val in b_flat] for a_val in a_flat]
