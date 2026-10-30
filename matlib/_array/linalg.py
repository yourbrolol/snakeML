"""Array-native linear algebra kernels operating on flat Buffers."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple, Union

from debug import get_logger
from debug.errors import ShapeError, ValidationError

from ._common import (
    array_from,
    coords_from_flat,
    flat_from_coords,
    flat_values,
)

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


def _scaled(values: List[Any], scalar: Any, shape: Tuple[int, ...]) -> Any:
    return array_from([value * scalar for value in values], shape)


def det(array: Any) -> Any:
    """Compute the determinant of a square matrix."""
    values, shape = flat_values(array)
    logger.debug("det requested", ndim=len(shape), shape=shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        logger.error("det only defined for square matrices", ndim=len(shape), shape=shape)
        raise ShapeError("determinant is only defined for square matrices")
    return _det_flat(values, shape[0])


def _det_flat(values: Sequence[Any], n: int) -> Any:
    if n == 0:
        return 0
    if n == 1:
        return values[0]
    if n == 2:
        return values[0] * values[3] - values[1] * values[2]

    determinant = 0
    for column in range(n):
        minor = [
            values[row * n + col]
            for row in range(1, n)
            for col in range(n)
            if col != column
        ]
        determinant += ((-1) ** column) * values[column] * _det_flat(minor, n - 1)
    return determinant


def inverse(array: Any) -> Any:
    """Compute the inverse of a 2x2 matrix."""
    values, shape = flat_values(array)
    logger.debug("inverse requested", ndim=len(shape), shape=shape)
    if shape != (2, 2):
        logger.error("inverse only implemented for 2x2 arrays", shape=shape)
        raise ShapeError("inverse is only implemented for 2x2 arrays")

    a, b, c, d = values
    det_value = a * d - b * c
    if det_value == 0:
        logger.error("matrix is singular", det=det_value)
        raise ShapeError("matrix is singular")

    return array_from(
        [d / det_value, -b / det_value, -c / det_value, a / det_value], (2, 2)
    )


def cross(a: Any, b: Any) -> Any:
    """Compute the cross product of two 3D vectors."""
    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)
    logger.debug("cross requested", a_shape=a_shape, b_shape=b_shape)
    if a_shape != (3,) or b_shape != (3,):
        logger.error("cross only defined for 3D vectors", a_shape=a_shape, b_shape=b_shape)
        raise ShapeError("cross product is only defined for 3D vectors")

    return array_from(
        [
            a_values[1] * b_values[2] - a_values[2] * b_values[1],
            a_values[2] * b_values[0] - a_values[0] * b_values[2],
            a_values[0] * b_values[1] - a_values[1] * b_values[0],
        ],
        (3,),
    )


def solve(a: Any, b: Any) -> Any:
    """Solve the linear system Ax = b for 2x2 matrices."""
    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)
    logger.debug("solve requested", a_shape=a_shape, b_shape=b_shape)
    if a_shape != (2, 2):
        logger.error("solve only implemented for 2x2 matrices", a_shape=a_shape)
        raise ShapeError("solve is only implemented for 2x2 arrays")

    a11, a12, a21, a22 = a_values
    det_a = a11 * a22 - a12 * a21
    if det_a == 0:
        logger.error("matrix is singular", det=det_a)
        raise ShapeError("matrix is singular")

    if len(b_shape) == 1 and len(b_values) >= 2:
        x1 = (a22 * b_values[0] - a12 * b_values[1]) / det_a
        x2 = (-a21 * b_values[0] + a11 * b_values[1]) / det_a
        return array_from([x1, x2], (2,))

    logger.error("solve only supports 1D right-hand side", b_shape=b_shape)
    raise ShapeError("solve currently only supports 1D or 2D right-hand side arrays")


def transpose(array: Any) -> Any:
    """Transpose a 2D array by swapping rows and columns."""
    values, shape = flat_values(array)
    logger.debug("transpose requested", ndim=len(shape), shape=shape)
    if len(shape) != 2:
        logger.error("transpose only defined for 2D arrays", ndim=len(shape))
        raise ShapeError("transpose is only defined for 2D arrays")

    rows, cols = shape
    result = [
        values[row * cols + col]
        for col in range(cols)
        for row in range(rows)
    ]
    return array_from(result, (cols, rows))


def permute(array: Any, axes: Sequence[int]) -> Any:
    """Permute the dimensions of an N-D array."""
    values, shape = flat_values(array)
    axes = list(axes)
    ndim = len(shape)

    if len(axes) != ndim:
        logger.error("permute axes length mismatch", ndim=ndim, axes=axes)
        raise ShapeError(f"permute axes must have length {ndim}, got {len(axes)}")
    if sorted(axes) != list(range(ndim)):
        logger.error("permute axes invalid", axes=axes)
        raise ShapeError(f"permute axes must be a permutation of 0..{ndim - 1}")

    new_shape = tuple(shape[axis] for axis in axes)
    result: List[Any] = []
    for out_index in range(_size(new_shape)):
        out_coords = coords_from_flat(out_index, new_shape)
        in_coords = [0] * ndim
        for out_axis, old_axis in enumerate(axes):
            in_coords[old_axis] = out_coords[out_axis]
        result.append(values[flat_from_coords(in_coords, shape)])

    logger.debug("permute completed", old_shape=shape, new_shape=new_shape, axes=axes)
    return array_from(result, new_shape)


def _size(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def tensordot(
    a: Any,
    b: Any,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
) -> Any:
    """Compute a tensor dot product along the requested axes."""
    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)

    axes_a: List[int]
    axes_b: List[int]
    if isinstance(axes, int):
        if axes < 0:
            logger.error("invalid tensordot axes", axes=axes)
            raise ValidationError("axes must be >= 0")
        axes_a = list(range(a_ndim - axes, a_ndim))
        axes_b = list(range(axes))
    else:
        axes_a = list(axes[0])
        axes_b = list(axes[1])

    axes_a = [axis % a_ndim for axis in axes_a]
    axes_b = [axis % b_ndim for axis in axes_b]

    if len(axes_a) != len(axes_b):
        logger.error("tensordot axes length mismatch", axes_a=axes_a, axes_b=axes_b)
        raise ShapeError("Axes lengths differ.")

    free_a = [axis for axis in range(a_ndim) if axis not in axes_a]
    free_b = [axis for axis in range(b_ndim) if axis not in axes_b]

    out_shape = tuple(a_shape[axis] for axis in free_a) + tuple(
        b_shape[axis] for axis in free_b
    )
    contract_shape = tuple(a_shape[axis] for axis in axes_a)

    if not out_shape:
        total = 0
        for contract_index in range(_size(contract_shape)):
            contract = coords_from_flat(contract_index, contract_shape)
            a_coords = [0] * a_ndim
            b_coords = [0] * b_ndim
            for axis, value in zip(axes_a, contract):
                a_coords[axis] = value
            for axis, value in zip(axes_b, contract):
                b_coords[axis] = value
            total += (
                a_values[flat_from_coords(a_coords, a_shape)]
                * b_values[flat_from_coords(b_coords, b_shape)]
            )
        return total

    result: List[Any] = []
    for out_index in range(_size(out_shape)):
        out_coords = coords_from_flat(out_index, out_shape)
        a_free = out_coords[: len(free_a)]
        b_free = out_coords[len(free_a) :]

        total = 0
        for contract_index in range(_size(contract_shape)):
            contract = coords_from_flat(contract_index, contract_shape)
            a_coords = [0] * a_ndim
            b_coords = [0] * b_ndim
            for axis, value in zip(free_a, a_free):
                a_coords[axis] = value
            for axis, value in zip(axes_a, contract):
                a_coords[axis] = value
            for axis, value in zip(free_b, b_free):
                b_coords[axis] = value
            for axis, value in zip(axes_b, contract):
                b_coords[axis] = value
            total += (
                a_values[flat_from_coords(a_coords, a_shape)]
                * b_values[flat_from_coords(b_coords, b_shape)]
            )
        result.append(total)

    return array_from(result, out_shape)


def matvec(a: Any, b: Any) -> Any:
    """Compute a matrix-vector product."""
    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)
    logger.debug("matvec requested", a_shape=a_shape, b_shape=b_shape)
    if len(a_shape) != 2 or len(b_shape) != 1 or a_shape[1] != b_shape[0]:
        logger.error("matvec shape mismatch", a_shape=a_shape, b_shape=b_shape)
        raise ShapeError(f"Shapes {a_shape} and {b_shape} not aligned.")

    rows, cols = a_shape
    result = [
        sum(a_values[row * cols + col] * b_values[col] for col in range(cols))
        for row in range(rows)
    ]
    return array_from(result, (rows,))


def matmul(a: Any, b: Any) -> Any:
    """Compute matrix multiplication or a dot product for 1D/2D arrays."""
    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)

    a_scalar = len(a_shape) == 0
    b_scalar = len(b_shape) == 0

    if a_scalar and b_scalar:
        return a_values[0] * b_values[0]
    if b_scalar:
        return _scaled(a_values, b_values[0], a_shape)
    if a_scalar:
        return _scaled(b_values, a_values[0], b_shape)

    if len(a_shape) == 1 and len(b_shape) == 1:
        return sum(x * y for x, y in zip(a_values, b_values))

    if len(a_shape) == 1 and len(b_shape) == 2:
        rows, cols = b_shape
        return array_from(
            [
                sum(
                    a_values[row] * b_values[row * cols + col]
                    for row in range(min(len(a_values), rows))
                )
                for col in range(cols)
            ],
            (cols,),
        )

    if len(a_shape) == 2 and len(b_shape) == 1:
        rows, cols = a_shape
        return array_from(
            [
                sum(
                    a_values[row * cols + col] * b_values[col]
                    for col in range(min(cols, len(b_values)))
                )
                for row in range(rows)
            ],
            (rows,),
        )

    if len(a_shape) == 2 and len(b_shape) == 2:
        rows, inner = a_shape
        inner_b, cols = b_shape
        if inner != inner_b:
            logger.error(
                "matmul matrix shapes not aligned", a_shape=a_shape, b_shape=b_shape
            )
            raise ShapeError(
                f"Matrix shapes not aligned: {inner} != {inner_b}"
            )
        return array_from(
            [
                sum(
                    a_values[row * inner + k] * b_values[k * cols + col]
                    for k in range(inner)
                )
                for row in range(rows)
                for col in range(cols)
            ],
            (rows, cols),
        )

    logger.error("matmul unsupported ranks", a_shape=a_shape, b_shape=b_shape)
    raise ShapeError(
        f"matmul is not implemented for shapes {a_shape} and {b_shape}"
    )


def outer(a: Any, b: Any) -> Any:
    """Compute the outer product of two flattened operands."""
    a_values, _ = flat_values(a)
    b_values, _ = flat_values(b)
    return array_from(
        [a_value * b_value for a_value in a_values for b_value in b_values],
        (len(a_values), len(b_values)),
    )


def cholesky(array: Any) -> Any:
    """Compute the Cholesky decomposition of a 2x2 positive-definite matrix."""
    values, shape = flat_values(array)
    logger.debug("cholesky requested", ndim=len(shape), shape=shape)
    if shape != (2, 2):
        logger.error("cholesky only implemented for 2x2 arrays", shape=shape)
        raise ShapeError("cholesky is only implemented for 2x2 arrays")

    a11, _, a21, a22 = values

    if a11 <= 0 or (a22 - (a21**2) / a11) <= 0:
        logger.error("matrix is not positive-definite", a11=a11, a21=a21, a22=a22)
        raise ShapeError("matrix is not positive-definite")

    l11 = math.sqrt(a11)
    l21 = a21 / l11
    l22 = math.sqrt(a22 - l21**2)

    return array_from([l11, 0, l21, l22], (2, 2))
