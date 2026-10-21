"""Elementwise and unary math operations for Array."""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from debug import get_logger
from debug.errors import ShapeError
from matlib import broadcasting, linalg
from matlib import ops as matlib_ops
from structs.dtypes import BinaryOp, Self

from .protocol import ArrayProtocol

logger = get_logger(__name__)

__all__ = ["ArrayMathOps"]


class ArrayMathOps(ArrayProtocol):
    """Mixin providing elementwise and unary math behavior."""

    def _elementwise(self, other: Any, op: BinaryOp) -> Self:
        """Applies a binary operator element-wise between self and another array or scalar.

        Shapes are broadcast against each other following NumPy rules before
        ``op`` is applied pairwise.

        Args:
            other: The right-hand operand: an ``Array``, a nested sequence, or a
                scalar broadcast across every element.
            op: Two-argument callable receiving the ``(self, other)`` pair of
                broadcast values, e.g. ``lambda x, y: x + y``.

        Returns:
            Self: A new array of class ``type(self)`` holding the results.

        Raises:
            ShapeError: If the two shapes cannot be broadcast together.
        """
        other_data = self._unwrap(other)
        return self.__class__(broadcasting.elementwise(self.data, other_data, op))

    def _update_self(self, result: Any) -> Self:
        """Rebuild shape metadata after ``data`` was replaced by a new value.

        Args:
            result: The value to adopt, normally an ``Array`` produced by an
                in-place operator such as ``__iadd__``.

        Returns:
            Self: ``self``, after ``data``/``shape``/``ndim`` have been refreshed.
            Note that ``strides`` is not recomputed.
        """
        self.data = result.data if hasattr(result, "data") else result
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    # ----------------------------------------------------------------- #
    # Arithmetic and comparison dunders
    # ----------------------------------------------------------------- #

    def __add__(self, other: Any) -> Self:
        """Element-wise sum of ``self + other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise sums.
        """
        return self._elementwise(other, lambda x, y: x + y)

    def __radd__(self, other: Any) -> Self:
        """Element-wise sum with ``other`` on the left, for ``other + self``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise sums; addition is commutative.
        """
        return self.__add__(other)

    def __sub__(self, other: Any) -> Self:
        """Element-wise difference of ``self - other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise differences.
        """
        return self._elementwise(other, lambda x, y: x - y)

    def __rsub__(self, other: Any) -> Self:
        """Element-wise difference with ``other`` on the left, for ``other - self``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise differences, operand order swapped.
        """
        return self._elementwise(other, lambda x, y: y - x)

    def __mul__(self, other: Any) -> Self:
        """Element-wise product of ``self * other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise products.
        """
        return self._elementwise(other, lambda x, y: x * y)

    def __rmul__(self, other: Any) -> Self:
        """Element-wise product with ``other`` on the left, for ``other * self``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array with the pairwise products; multiplication is
            commutative.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Self:
        """Element-wise true division of ``self / other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar divisor.

        Returns:
            Self: A new array with the pairwise quotients.

        Raises:
            ZeroDivisionError: If any divisor element is ``0``.
        """
        return self._elementwise(other, lambda x, y: x / y)

    def __rtruediv__(self, other: Any) -> Self:
        """Element-wise division with ``other`` on the left, for ``other / self``.

        Args:
            other: ``Array``, nested sequence, or scalar dividend.

        Returns:
            Self: A new array with the pairwise quotients, operand order swapped.
        """
        return self._elementwise(other, lambda x, y: y / x)

    def __floordiv__(self, other: Any) -> Self:
        """Element-wise floor division of ``self // other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar divisor.

        Returns:
            Self: A new array with the pairwise floor quotients.

        Raises:
            ZeroDivisionError: If any divisor element is ``0``.
        """
        return self._elementwise(other, lambda x, y: x // y)

    def __rfloordiv__(self, other: Any) -> Self:
        """Element-wise floor division with ``other`` on the left, for ``other // self``.

        Args:
            other: ``Array``, nested sequence, or scalar dividend.

        Returns:
            Self: A new array with the pairwise floor quotients, order swapped.
        """
        return self._elementwise(other, lambda x, y: y // x)

    def __mod__(self, other: Any) -> Self:
        """Element-wise remainder of ``self % other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar modulus.

        Returns:
            Self: A new array with the pairwise remainders.

        Raises:
            ZeroDivisionError: If any modulus element is ``0``.
        """
        return self._elementwise(other, lambda x, y: x % y)

    def __rmod__(self, other: Any) -> Self:
        """Element-wise remainder with ``other`` on the left, for ``other % self``.

        Args:
            other: ``Array``, nested sequence, or scalar dividend.

        Returns:
            Self: A new array with the pairwise remainders, operand order swapped.
        """
        return self._elementwise(other, lambda x, y: y % x)

    def __neg__(self) -> Self:
        """Element-wise negation, written as ``-1 * self``.

        Returns:
            Self: A new array with every element negated.
        """
        return -1 * self

    def __gt__(self, other: Any) -> Self:
        """Element-wise ``self > other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x > y)

    def __lt__(self, other: Any) -> Self:
        """Element-wise ``self < other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x < y)

    def __ge__(self, other: Any) -> Self:
        """Element-wise ``self >= other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x >= y)

    def __le__(self, other: Any) -> Self:
        """Element-wise ``self <= other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x <= y)

    def __eq__(self, other: Any) -> Any:
        """Element-wise ``self == other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x == y)

    def __ne__(self, other: Any) -> Any:
        """Element-wise ``self != other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: A new array of booleans; ``False`` where the test fails.
        """
        return self._elementwise(other, lambda x, y: x != y)

    def __and__(self, other: Any) -> Self:
        """Element-wise logical ``and`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar operand.

        Returns:
            Self: A new array of the logical ``and`` of each pair. Operands
            are returned as-is rather than coerced to ``bool``.
        """
        return self._elementwise(other, lambda x, y: x and y)

    def __or__(self, other: Any) -> Self:
        """Element-wise logical ``or`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar operand.

        Returns:
            Self: A new array of the logical ``or`` of each pair.
        """
        return self._elementwise(other, lambda x, y: x or y)

    def __xor__(self, other: Any) -> Self:
        """Element-wise bitwise ``^`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar operand.

        Returns:
            Self: A new array of the pairwise exclusive-or.

        Raises:
            TypeError: If any element pair does not support ``^`` (e.g. floats).
        """
        return self._elementwise(other, lambda x, y: x ^ y)

    def __abs__(self) -> Self:
        """Element-wise absolute value, via ``abs(self)``.

        Returns:
            Self: A new array with the magnitude of every element.
        """
        return self._wrap_result(matlib_ops.abs(self))

    # ----------------------------------------------------------------- #
    # In-place arithmetic dunders
    # ----------------------------------------------------------------- #

    def __iadd__(self, other: Any) -> Self:
        """In-place element-wise ``+=``; mutates and rebinds ``data``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: ``self``, updated in place.
        """
        return self._update_self(self._elementwise(other, lambda x, y: x + y))

    def __isub__(self, other: Any) -> Self:
        """In-place element-wise ``-=``; mutates and rebinds ``data``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: ``self``, updated in place.
        """
        return self._update_self(self._elementwise(other, lambda x, y: x - y))

    def __imul__(self, other: Any) -> Self:
        """In-place element-wise ``*=``; mutates and rebinds ``data``.

        Args:
            other: ``Array``, nested sequence, or scalar.

        Returns:
            Self: ``self``, updated in place.
        """
        return self._update_self(self._elementwise(other, lambda x, y: x * y))

    def __itruediv__(self, other: Any) -> Self:
        """In-place element-wise ``/=``; mutates and rebinds ``data``.

        Args:
            other: ``Array``, nested sequence, or scalar divisor.

        Returns:
            Self: ``self``, updated in place.

        Raises:
            ZeroDivisionError: If any divisor element is ``0``.
        """
        return self._update_self(self._elementwise(other, lambda x, y: x / y))

    def __ipow__(self, other: Any) -> Self:
        """In-place element-wise ``**=``; mutates and rebinds ``data``.

        Args:
            other: ``Array``, nested sequence, or scalar exponent.

        Returns:
            Self: ``self``, updated in place.
        """
        return self._update_self(self._elementwise(other, lambda x, y: x**y))

    def __imatmul__(self, other: Any) -> Self:
        """In-place matrix multiplication ``@=``.

        Args:
            other: Right-hand ``Array`` operand.

        Returns:
            Self: ``self``, updated in place with the product.
        """
        return self._update_self(self.matmul(other))

    # ----------------------------------------------------------------- #
    # Linear algebra
    # ----------------------------------------------------------------- #

    def dot(self, other: Any, axes: Union[int, tuple] = 1) -> Self:
        """Generalised dot product over the requested axes.

        Args:
            other: Right-hand ``Array``.
            axes: Number of trailing axes of ``self`` contracted against the
                leading axes of ``other``, e.g. ``1`` for matrix-vector style
                contraction.

        Returns:
            Self: A new array with the contracted axes removed.

        Raises:
            ShapeError: If the contracted dimensions do not match.
        """
        return self._wrap_result(linalg.tensordot(self, other, axes=axes))

    def inner(self, other: Any) -> Self:
        """Inner product: the sum of element-wise products.

        Args:
            other: ``Array``. When both operands are 1-D their lengths must
                match exactly; otherwise broadcasting rules apply.

        Returns:
            Self: A new array holding the contraction over the last axis of
            ``self`` and the first axis of ``other``.

        Raises:
            ShapeError: If two 1-D operands have different shapes, or the
                contracted dimensions are otherwise incompatible.
        """
        if self.ndim == 1 and getattr(other, "ndim", 0) == 1:
            if self.shape != other.shape:
                raise ShapeError(
                    f"inner: incompatible shapes {self.shape} and {other.shape}"
                )

        return self.dot(other, axes=1)

    def matvec(self, other: Any) -> Self:
        """Matrix-vector product.

        Args:
            other: ``Array`` treated as a 1-D vector.

        Returns:
            Self: A new array of shape ``(m,)`` for an ``(m, n)`` input.

        Raises:
            ShapeError: If the inner dimensions do not match.
        """
        return self._wrap_result(linalg.matvec(self, other))

    def matmul(self, other: Any) -> Self:
        """Matrix multiplication, broadcasting over any leading batch axes.

        Args:
            other: ``Array`` with rank >= 1 whose trailing dimensions are
                compatible with the leading dimensions of ``self``.

        Returns:
            Self: A new array holding the batched matrix product.

        Raises:
            ShapeError: If the contracted dimensions disagree.
        """
        return self._wrap_result(linalg.matmul(self, other))

    def outer(self, other: Any) -> Self:
        """Outer product of the flattened operands.

        Args:
            other: ``Array`` of any shape; it is flattened before use.

        Returns:
            Self: A new 2-D array of shape ``(self.size, other.size)``.

        Raises:
            ShapeError: If either operand is 0-d.
        """
        return self._wrap_result(linalg.outer(self, other))

    def __matmul__(self, other: Any) -> Self:
        """Dispatches ``self @ other`` to :meth:`matmul`.

        Args:
            other: Right-hand ``Array``.

        Returns:
            Self: The matrix product; see :meth:`matmul`.
        """
        return self.matmul(other)

    def __pow__(self, other: Any) -> Self:
        """Element-wise ``self ** other`` with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar exponent.

        Returns:
            Self: A new array with each base raised to the paired exponent.
        """
        return self._elementwise(other, lambda x, y: x**y)

    # ----------------------------------------------------------------- #
    # Unary math functions
    # ----------------------------------------------------------------- #

    def exp(self) -> Self:
        """Element-wise natural exponential.

        Returns:
            Self: A new array of ``e ** x`` values, shape unchanged.
        """
        return self._wrap_result(matlib_ops.exp(self))

    def log(self) -> Self:
        """Element-wise natural logarithm.

        Returns:
            Self: A new array of ``ln(x)`` values.

        Raises:
            ValueError: If any element is negative.
            ZeroDivisionError: If any element is ``0``.
        """
        return self._wrap_result(matlib_ops.log(self))

    def log2(self) -> Self:
        """Element-wise base-2 logarithm.

        Returns:
            Self: A new array of ``log2(x)`` values.

        Raises:
            ValueError: If any element is negative.
            ZeroDivisionError: If any element is ``0``.
        """
        return self._wrap_result(matlib_ops.log2(self))

    def log10(self) -> Self:
        """Element-wise base-10 logarithm.

        Returns:
            Self: A new array of ``log10(x)`` values.

        Raises:
            ValueError: If any element is negative.
            ZeroDivisionError: If any element is ``0``.
        """
        return self._wrap_result(matlib_ops.log10(self))

    def sqrt(self) -> Self:
        """Element-wise square root.

        Returns:
            Self: A new array of square roots.

        Raises:
            ValueError: If any element is negative.
        """
        return self._wrap_result(matlib_ops.sqrt(self))

    def square(self) -> Self:
        """Element-wise squaring.

        Returns:
            Self: A new array with every element multiplied by itself.
        """
        return self._wrap_result(matlib_ops.square(self))

    def abs(self) -> Self:
        """Element-wise absolute value.

        Returns:
            Self: A new array of magnitudes; the same result as ``abs(self)``.
        """
        return self._wrap_result(matlib_ops.abs(self))

    def sin(self) -> Self:
        """Element-wise sine.

        Returns:
            Self: A new array of ``sin(x)`` values.
        """
        return self._wrap_result(matlib_ops.sin(self))

    def cos(self) -> Self:
        """Element-wise cosine.

        Returns:
            Self: A new array of ``cos(x)`` values.
        """
        return self._wrap_result(matlib_ops.cos(self))

    def tan(self) -> Self:
        """Element-wise tangent.

        Returns:
            Self: A new array of ``tan(x)`` values.
        """
        return self._wrap_result(matlib_ops.tan(self))

    def asin(self) -> Self:
        """Element-wise inverse sine.

        Returns:
            Self: A new array of ``asin(x)`` values.

        Raises:
            ValueError: If any element lies outside ``[-1, 1]``.
        """
        return self._wrap_result(matlib_ops.asin(self))

    def acos(self) -> Self:
        """Element-wise inverse cosine.

        Returns:
            Self: A new array of ``acos(x)`` values.

        Raises:
            ValueError: If any element lies outside ``[-1, 1]``.
        """
        return self._wrap_result(matlib_ops.acos(self))

    def atan(self) -> Self:
        """Element-wise inverse tangent.

        Returns:
            Self: A new array of ``atan(x)`` values in ``(-pi/2, pi/2)``.
        """
        return self._wrap_result(matlib_ops.atan(self))

    def atan2(self, other: Any) -> Self:
        """Element-wise two-argument inverse tangent with broadcasting.

        Args:
            other: ``Array``, nested sequence, or scalar providing the ``y``
                argument; ``self`` supplies ``x``.

        Returns:
            Self: A new array of ``atan2(x, y)`` values in ``(-pi, pi]``.
        """
        return self._elementwise(other, lambda x, y: math.atan2(x, y))

    def sinh(self) -> Self:
        """Element-wise hyperbolic sine.

        Returns:
            Self: A new array of ``sinh(x)`` values.
        """
        return self._wrap_result(matlib_ops.sinh(self))

    def cosh(self) -> Self:
        """Element-wise hyperbolic cosine.

        Returns:
            Self: A new array of ``cosh(x)`` values.
        """
        return self._wrap_result(matlib_ops.cosh(self))

    def tanh(self) -> Self:
        """Element-wise hyperbolic tangent.

        Returns:
            Self: A new array of ``tanh(x)`` values.
        """
        return self._wrap_result(matlib_ops.tanh(self))

    def floor(self) -> Self:
        """Element-wise floor (round toward negative infinity).

        Returns:
            Self: A new array of integral floors, dtype following the input.
        """
        return self._wrap_result(matlib_ops.floor(self))

    def ceil(self) -> Self:
        """Element-wise ceiling (round toward positive infinity).

        Returns:
            Self: A new array of integral ceilings, dtype following the input.
        """
        return self._wrap_result(matlib_ops.ceil(self))

    def round(self, ndigits: Optional[int] = None) -> Self:
        """Element-wise rounding to the nearest value.

        Args:
            ndigits: Number of decimal places to keep. ``None`` (default)
                rounds to a whole number, and ties round to even.

        Returns:
            Self: A new array of rounded values.
        """
        return self._wrap_result(matlib_ops.round(self, ndigits))

    def clip(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> Self:
        """Element-wise clamping to the given bounds.

        Args:
            min_value: Lower bound. Values below it become this; ``None``
                leaves the lower side untouched.
            max_value: Upper bound. Values above it become this; ``None``
                leaves the upper side untouched.

        Returns:
            Self: A new array with every element inside the bounds.

        Raises:
            ValueError: If both bounds are ``None``.
        """
        return self._wrap_result(
            matlib_ops.clip(self, min_value=min_value, max_value=max_value)
        )

    def sign(self) -> Self:
        """Element-wise sign.

        Returns:
            Self: A new array of ``-1``, ``0`` or ``1`` per element.
        """
        return self._wrap_result(matlib_ops.sign(self))

    # ----------------------------------------------------------------- #
    # Structural reductions
    # ----------------------------------------------------------------- #

    def trace(self) -> Any:
        """Sum of the main-diagonal entries of a 2-D array.

        Returns:
            Any: The scalar sum of the diagonal.

        Raises:
            ShapeError: If ``ndim`` is not ``2``.
        """
        if self.ndim != 2:
            raise ShapeError("trace requires a 2D array")
        return sum(self[i][i] for i in range(min(self.shape)))

    def diag(self) -> Self:
        """Diagonal matrix from a vector, or diagonal vector from a matrix.

        Returns:
            Self: For 1-D ``self`` a 2-D square matrix whose main diagonal is
            ``self``; for 2-D ``self`` the 1-D main diagonal.

        Raises:
            ShapeError: If ``ndim`` is neither ``1`` nor ``2``.
        """
        if self.ndim == 1:
            return self.__class__(
                [
                    [value if i == j else 0 for j in range(len(self))]
                    for i, value in enumerate(self)
                ]
            )
        if self.ndim == 2:
            return self.__class__([self[i][i] for i in range(min(self.shape))])
        raise ShapeError("diag only supports 1D or 2D arrays")

    def diagonal(self) -> Self:
        """Main diagonal of a 2-D array.

        Returns:
            Self: A new 1-D array of length ``min(shape)``.

        Raises:
            ShapeError: If ``ndim`` is not ``2``.
        """
        if self.ndim != 2:
            raise ShapeError("diagonal requires a 2D array")
        return self.__class__([self[i][i] for i in range(min(self.shape))])

    # ----------------------------------------------------------------- #
    # Decompositions and solvers
    # ----------------------------------------------------------------- #

    def inverse(self) -> Self:
        """Matrix inverse of a square 2-D array.

        Returns:
            Self: A new array ``A`` such that ``A @ A_inv`` is the identity.

        Raises:
            ShapeError: If the input is not square.
            ZeroDivisionError: If the matrix is singular.
        """
        return self._wrap_result(linalg.inverse(self))

    def det(self) -> Any:
        """Determinant of a square 2-D array.

        Returns:
            Any: The scalar determinant.

        Raises:
            ShapeError: If the input is not 2-D and square.
        """
        return linalg.det(self)

    def solve(self, other: Any) -> Any:
        """Solve the linear system ``self @ x = other``.

        Args:
            other: ``Array`` holding the right-hand side vector or matrix.

        Returns:
            Any: The solution ``x``; an ``Array`` for vector right-hand sides.

        Raises:
            ShapeError: If the dimensions are incompatible.
            ZeroDivisionError: If ``self`` is singular.
        """
        return linalg.solve(self, other)

    def pinv(self) -> Self:
        """Pseudo-inverse of a 2-D array.

        Returns:
            Self: A new array approximating the Moore-Penrose inverse; this
            currently delegates to the exact :meth:`inverse` and therefore
            requires a square, non-singular input.
        """
        return linalg.inverse(self)

    def cholesky(self) -> Any:
        """Cholesky decomposition ``A = L @ L.T``.

        Returns:
            Any: The lower-triangular factor ``L``.

        Raises:
            ShapeError: If the input is not square or symmetric.
            ValueError: If the input is not positive definite.
        """
        return linalg.cholesky(self)

    def cross(self, other: Any) -> Any:
        """Cross product of two vectors (or arrays of stacked vectors).

        Args:
            other: ``Array`` with a matching size-3 final dimension.

        Returns:
            Any: The cross product along the last axis.

        Raises:
            ShapeError: If the operands are not size 3 or their shapes differ.
        """
        return linalg.cross(self, other)

    # ----------------------------------------------------------------- #
    # Not yet implemented
    # ----------------------------------------------------------------- #

    def svd(self) -> Self:
        """Singular value decomposition.

        Returns:
            Self: The decomposition factors, once implemented.

        Raises:
            NotImplementedError: Always; not implemented yet.
        """
        raise NotImplementedError("svd is not implemented yet")

    def qr(self) -> Self:
        """QR decomposition.

        Returns:
            Self: The factors ``Q`` and ``R``, once implemented.

        Raises:
            NotImplementedError: Always; not implemented yet.
        """
        raise NotImplementedError("qr is not implemented yet")

    def eig(self) -> Self:
        """Eigenvalue decomposition.

        Returns:
            Self: The eigenvalues and eigenvectors, once implemented.

        Raises:
            NotImplementedError: Always; not implemented yet.
        """
        raise NotImplementedError("eig is not implemented yet")

    def eigh(self) -> Self:
        """Hermitian eigenvalue decomposition.

        Returns:
            Self: The eigenvalues and eigenvectors, once implemented.

        Raises:
            NotImplementedError: Always; not implemented yet.
        """
        raise NotImplementedError("eigh is not implemented yet")


__all__ = ["ArrayMathOps"]