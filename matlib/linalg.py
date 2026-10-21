"""Linear algebra: determinants, inverses, products and decompositions.

Operates on :class:`~structs.array.array.Array` objects (element access
via ``__getitem__``) exposing ``.shape``/``.ndim`` and returns nested
lists. All shape violations raise :class:`debug.errors.ShapeError` and
unsupported inputs raise :class:`debug.errors.ValidationError`.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple, Union

from debug import get_logger
from debug.errors import ShapeError, ValidationError
from structs.utils import build_index, indices, set_nested, zeroes

logger = get_logger(__name__)

__all__ = [
    "cholesky",
    "cross",
    "det",
    "inverse",
    "matmul",
    "matvec",
    "outer",
    "permute",
    "solve",
    "tensordot",
    "transpose",
]


def det(array: Any) -> Any:
    """Compute the determinant of a square matrix.

    Uses Laplace expansion along the first row (cofactor recursion).

    Args:
        array: Square 2-D array.

    Returns:
        The scalar determinant.

    Raises:
        ShapeError: If ``array`` is not square.
    """
    logger.debug("det requested", ndim=array.ndim, shape=getattr(array, "shape", None))
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        logger.error(
            "det only defined for square matrices",
            ndim=array.ndim,
            shape=getattr(array, "shape", None),
        )
        raise ShapeError("determinant is only defined for square matrices")

    n = array.shape[0]
    if n == 1:
        return array[0][0]
    elif n == 2:
        return array[0][0] * array[1][1] - array[0][1] * array[1][0]

    determinant = 0
    for c in range(n):
        minor = [[array[r][cc] for cc in range(n) if cc != c] for r in range(1, n)]
        determinant += ((-1) ** c) * array[0][c] * det(minor)

    return determinant


def inverse(array: Any) -> List[List[float]]:
    """Compute the inverse of a 2x2 matrix.

    Args:
        array: Matrix of shape ``(2, 2)``.

    Returns:
        The inverse as a nested list of floats.

    Raises:
        ShapeError: If ``array`` is not ``(2, 2)`` or is singular.
    """
    logger.debug(
        "inverse requested", ndim=array.ndim, shape=getattr(array, "shape", None)
    )
    if array.shape != (2, 2):
        logger.error(
            "inverse only implemented for 2x2 arrays",
            shape=getattr(array, "shape", None),
        )
        raise ShapeError("inverse is only implemented for 2x2 arrays")

    a, b = array[0][0], array[0][1]
    c, d = array[1][0], array[1][1]
    det_value = det(array)

    if det_value == 0:
        logger.error("matrix is singular", det=det_value)
        raise ShapeError("matrix is singular")

    return [[d / det_value, -b / det_value], [-c / det_value, a / det_value]]


def cross(a: Any, b: Any) -> List[Any]:
    """Compute the cross product of two 3D vectors.

    Args:
        a: Left vector; must be 1-D of length 3.
        b: Right vector; must be 1-D of length 3.

    Returns:
        The 3-element cross product as a list.

    Raises:
        ShapeError: If either vector is not 1-D of length 3.
    """
    logger.debug(
        "cross requested",
        a_shape=getattr(a, "shape", None),
        b_shape=getattr(b, "shape", None),
    )
    if a.ndim != 1 or b.ndim != 1 or len(a) != 3 or len(b) != 3:
        logger.error(
            "cross only defined for 3D vectors",
            a_shape=getattr(a, "shape", None),
            b_shape=getattr(b, "shape", None),
        )
        raise ShapeError("cross product is only defined for 3D vectors")

    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def solve(a: Any, b: Any) -> List[float]:
    """Solve the linear system Ax = b for 2x2 matrices.

    Args:
        a: Coefficient matrix of shape ``(2, 2)``.
        b: Right-hand side; must be a 1-D vector of length 2.

    Returns:
        The solution vector ``x`` as a two-element list.

    Raises:
        ShapeError: If ``a`` is not ``(2, 2)``, is singular, or ``b`` is
            not 1-D.
    """
    logger.debug(
        "solve requested",
        a_shape=getattr(a, "shape", None),
        b_shape=getattr(b, "shape", None),
    )
    if a.shape != (2, 2):
        logger.error("solve only implemented for 2x2 matrices", a_shape=a.shape)
        raise ShapeError("solve is only implemented for 2x2 arrays")

    a11, a12 = a[0][0], a[0][1]
    a21, a22 = a[1][0], a[1][1]
    det_a = det(a)

    if det_a == 0:
        logger.error("matrix is singular", det=det_a)
        raise ShapeError("matrix is singular")

    b_data = b.data if hasattr(b, "data") else b
    if isinstance(b_data, list) and not isinstance(b_data[0], list):
        x1 = (a22 * b_data[0] - a12 * b_data[1]) / det_a
        x2 = (-a21 * b_data[0] + a11 * b_data[1]) / det_a
        return [x1, x2]

    logger.error(
        "solve only supports 1D or 2D right-hand side",
        b_shape=getattr(b, "shape", None),
    )
    raise ShapeError("solve currently only supports 1D or 2D right-hand side arrays")


def transpose(array: Any) -> List[List[Any]]:
    """Transpose a 2D array by swapping rows and columns.

    Args:
        array: 2-D array (or nested list) of shape ``(rows, cols)``.

    Returns:
        The transposed nested list of shape ``(cols, rows)``.

    Raises:
        ShapeError: If ``array`` is not 2-D.
    """
    logger.debug(
        "transpose requested", ndim=array.ndim, shape=getattr(array, "shape", None)
    )
    if array.ndim != 2:
        logger.error("transpose only defined for 2D arrays", ndim=array.ndim)
        raise ShapeError("transpose is only defined for 2D arrays")

    rows = len(array)
    cols = len(array[0]) if rows else 0

    return [[array[r][c] for r in range(rows)] for c in range(cols)]


def permute(array: Any, axes: Sequence[int]) -> Any:
    """Permute the dimensions of an N-D array according to the given axes order.

    Parameters
    ----------
    array : Array
        The input array.
    axes : tuple or list of int
        The desired axis ordering, e.g. (2, 0, 1) to move axis 2 first.

    Returns:
        A nested list with dimensions reordered per ``axes``.

    Raises:
        ShapeError: If ``axes`` is not a permutation of ``0..ndim-1``.
    """
    axes = list(axes)
    ndim = array.ndim
    shape = array.shape

    if len(axes) != ndim:
        logger.error("permute axes length mismatch", ndim=ndim, axes=axes)
        raise ShapeError(f"permute axes must have length {ndim}, got {len(axes)}")
    if sorted(axes) != list(range(ndim)):
        logger.error("permute axes invalid", axes=axes)
        raise ShapeError(f"permute axes must be a permutation of 0..{ndim - 1}")

    new_shape = tuple(shape[ax] for ax in axes)
    result = zeroes(new_shape)

    for old_idx in indices(shape):
        new_idx = tuple(old_idx[ax] for ax in axes)
        set_nested(result, new_idx, array[old_idx])

    logger.debug("permute completed", old_shape=shape, new_shape=new_shape, axes=axes)
    return result


def tensordot(
    a: Any,
    b: Any,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
) -> Any:
    """Compute tensor dot product along specified contraction axes.

    Args:
        a: Left tensor.
        b: Right tensor.
        axes: Either an ``int`` (contract the last ``axes`` dims of ``a``
            with the first ``axes`` dims of ``b``) or a pair
            ``(axes_a, axes_b)`` of per-tensor axis lists.

    Returns:
        The contracted nested list, or a plain scalar when both tensors
        are fully contracted.

    Raises:
        ShapeError: If the two contraction-axis lists differ in length.
        ValidationError: If an integer ``axes`` argument is negative.
    """
    axes_a: List[int]
    axes_b: List[int]
    if isinstance(axes, int):
        if axes < 0:
            logger.error("invalid tensordot axes", axes=axes)
            raise ValidationError("axes must be >= 0")

        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(axes))
    else:
        axes_a = list(axes[0])
        axes_b = list(axes[1])

    axes_a = [ax % a.ndim for ax in axes_a]
    axes_b = [ax % b.ndim for ax in axes_b]

    if len(axes_a) != len(axes_b):
        logger.error("tensordot axes length mismatch", axes_a=axes_a, axes_b=axes_b)
        raise ShapeError("Axes lengths differ.")

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
        a_free = out_idx[: len(free_a)]
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


def matvec(a: Any, b: Any) -> List[Any]:
    """Compute matrix-vector product.

    Args:
        a: Matrix of shape ``(rows, cols)``.
        b: Vector of shape ``(cols,)``.

    Returns:
        The product vector as a list of length ``rows``.

    Raises:
        ShapeError: If ``a.shape[1] != b.shape[0]``.
    """
    logger.debug(
        "matvec requested",
        a_shape=getattr(a, "shape", None),
        b_shape=getattr(b, "shape", None),
    )
    if a.shape[1] != b.shape[0]:
        logger.error("matvec shape mismatch", a_shape=a.shape, b_shape=b.shape)
        raise ShapeError(f"Shapes {a.shape} and {b.shape} not aligned.")

    result = []
    for row in a:
        result.append(sum(r * v for r, v in zip(row, b)))
    return result


def matmul(a: Any, b: Any) -> Any:
    """Compute matrix multiplication or dot product for 1D/2D arrays.

    Dispatches on the operand ranks: scalar×scalar, 1D×scalar,
    1D·1D (inner product), 1D×2D, 2D×1D and 2D×2D, unwrapping ``.data``
    from array-like operands first.

    Args:
        a: Left operand (array-like, nested list or scalar).
        b: Right operand in the same accepted forms.

    Returns:
        The product in the natural shape of the input combination.

    Raises:
        ShapeError: If the inner dimensions of two matrix operands
            mismatch.
    """
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
        logger.error(
            "matmul matrix shapes not aligned",
            a_shape=getattr(a, "shape", None),
            b_shape=getattr(b, "shape", None),
        )
        raise ShapeError(
            f"Matrix shapes not aligned: {len(a_data[0])} != {len(b_data)}"
        )

    return [
        [
            sum(row[k] * b_data[k][col_idx] for k in range(len(b_data)))
            for col_idx in range(len(b_data[0]))
        ]
        for row in a_data
    ]


def outer(a: Any, b: Any) -> List[List[Any]]:
    """Compute outer product of two 1D arrays or flattened sequences.

    Any operand wrapped in a ``.data``-bearing array is unwrapped, and
    non-1-D data is flattened row-major before multiplication.

    Args:
        a: Left 1-D (or flappable) operand.
        b: Right 1-D (or flappable) operand.

    Returns:
        Nested list of shape ``(len(a_flat), len(b_flat))``.
    """
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


def cholesky(array: Any) -> List[List[float]]:
    """Compute the Cholesky decomposition of a 2x2 positive-definite matrix.

    Args:
        array: Symmetric positive-definite matrix of shape ``(2, 2)``.

    Returns:
        Lower-triangular ``L`` as a nested list with ``A = L @ L.T``.

    Raises:
        ShapeError: If ``array`` is not ``(2, 2)`` or is not
            positive-definite.
    """
    logger.debug(
        "cholesky requested", ndim=array.ndim, shape=getattr(array, "shape", None)
    )
    if array.shape != (2, 2):
        logger.error(
            "cholesky only implemented for 2x2 arrays",
            shape=getattr(array, "shape", None),
        )
        raise ShapeError("cholesky is only implemented for 2x2 arrays")

    a11 = array[0][0]
    a21 = array[1][0]
    a22 = array[1][1]

    if a11 <= 0 or (a22 - (a21**2) / a11) <= 0:
        logger.error("matrix is not positive-definite", a11=a11, a21=a21, a22=a22)
        raise ShapeError("matrix is not positive-definite")

    l11 = math.sqrt(a11)
    l21 = a21 / l11
    l22 = math.sqrt(a22 - l21**2)

    return [[l11, 0], [l21, l22]]
