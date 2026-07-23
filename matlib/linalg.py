from debug import get_logger
from debug.errors import ShapeError, ValidationError
from structs.utils import build_index, zeroes, set_nested, indices

logger = get_logger(__name__)


def transpose(array):
    """Transpose a 2D array by swapping rows and columns."""
    logger.debug("transpose requested", ndim=array.ndim, shape=getattr(array, 'shape', None))
    if array.ndim != 2:
        logger.error("transpose only defined for 2D arrays", ndim=array.ndim)
        raise ShapeError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0]) if rows else 0

    return [[array[r][c] for r in range(rows)] for c in range(cols)]


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified contraction axes."""
    if isinstance(axes, int):
        if axes < 0:
            logger.error("invalid tensordot axes", axes=axes)
            raise ValidationError("axes must be >= 0")

        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
        axes_a = list(axes_a)
        axes_b = list(axes_b)

    axes_a = [ax % a.ndim for ax in axes_a]
    axes_b = [ax % b.ndim for ax in axes_b]

    if len(axes_a) != len(axes_b):
        logger.error("tensordot axes length mismatch", axes_a=axes_a, axes_b=axes_b)
        raise ShapeError("Axes lengths differ.")

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
    """Compute matrix-vector product."""
    logger.debug("matvec requested", a_shape=getattr(a, 'shape', None), b_shape=getattr(b, 'shape', None))
    if a.shape[1] != b.shape[0]:
        logger.error("matvec shape mismatch", a_shape=a.shape, b_shape=b.shape)
        raise ShapeError(f"Shapes {a.shape} and {b.shape} not aligned.")

    result = []
    for row in a:
        result.append(sum(r * v for r, v in zip(row, b)))
    return result


def matmul(a, b):
    """Compute matrix multiplication or dot product for 1D/2D arrays."""
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
        logger.error("matmul matrix shapes not aligned", a_shape=getattr(a, 'shape', None), b_shape=getattr(b, 'shape', None))
        raise ShapeError(f"Matrix shapes not aligned: {len(a_data[0])} != {len(b_data)}")

    return [
        [sum(row[k] * b_data[k][col_idx] for k in range(len(b_data))) for col_idx in range(len(b_data[0]))]
        for row in a_data
    ]


def outer(a, b):
    """Compute outer product of two 1D arrays or flattened sequences."""
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
