"""Activation layers for snakeML neural networks.

Every activation derives from :class:`Activation` (itself a
:class:`basic.layers.Layer`) and exposes ``forward``/``backward`` pairs
operating element-wise on :class:`~structs.Array` inputs.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Any, Optional, Union

from basic.layers import Layer
from debug import get_logger
from debug.errors import ShapeError
from matlib import exp
from structs import Array

logger = get_logger(__name__)

__all__ = [
    "Activation",
    "GELU",
    "LinearActivation",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
]


class Activation(Layer):
    """Base class for activation layers.

    Attributes:
        name: Human-readable activation label.
        input: Cached forward-pass input used by ``backward``.
        output: Cached forward-pass output (when the backward pass needs it).
    """

    def __init__(self, name: str) -> None:
        """Base activation layer initializing name and input state."""
        super().__init__()
        self.name = name
        self.input: Optional[Any] = None
        self.output: Optional[Any] = None

    def forward(self, x: Any) -> Any:
        raise NotImplementedError

    def backward(self, x: Any) -> Any:
        raise NotImplementedError


class LinearActivation(Activation):
    """Identity activation layer."""

    def __init__(self) -> None:
        super().__init__("LinearActivation")

    def forward(self, x: Any) -> Any:
        """Pass input through without transformation."""
        logger.debug("linear activation forward", activation=self.name, value=x)
        return x

    def backward(self, x: Any) -> Any:
        """Pass gradient through without transformation."""
        logger.debug("linear activation backward", activation=self.name, value=x)
        return x


class ReLU(Activation):
    """Rectified Linear Unit activation layer."""

    def __init__(self) -> None:
        super().__init__("ReLU")

    def forward(self, x: Any) -> Any:
        """Apply ReLU activation element-wise.

        Args:
            x: Input value or :class:`~structs.Array`.

        Returns:
            ``max(0, x)`` wrapped back into an ``Array`` if scalar.
        """
        self.input = x
        logger.debug("relu forward", value=x)
        out = max(0, x)
        return out if isinstance(out, Array) else Array([out])

    def backward(self, grad: Any) -> Any:
        """Compute backward pass for ReLU activation.

        Args:
            grad: Upstream gradient.

        Returns:
            ``grad`` masked by ``input > 0``.
        """
        logger.debug("relu backward", grad=grad, input=self.input)
        return grad * (self.input > 0)


class GELU(Activation):
    """Gaussian Error Linear Unit activation layer."""

    def __init__(self) -> None:
        super().__init__("GELU")

    def forward(self, x: Any) -> Any:
        """Apply GELU activation element-wise.

        Args:
            x: Input :class:`~structs.Array`.

        Returns:
            GELU output, same shape as the input.
        """
        self.input = x
        logger.debug("gelu forward", value=x)
        out = 0.5 * x * (1 + (sqrt(2 / pi) * (x + 0.044715 * x**3)).tanh())
        return out if isinstance(out, Array) else Array([out])

    def backward(self, grad: Any) -> Any:
        """Compute backward pass for GELU activation.

        Uses the analytic derivative of the tanh approximation of GELU:
        ``0.5 * (1 + tanh(u)) + 0.5 * x * sech2(u) * k * (1 + 0.134145 * x**2)``.

        Args:
            grad: Upstream gradient.

        Returns:
            ``grad * dGELU/dx`` w.r.t. the cached input.
        """
        logger.debug("gelu backward", grad=grad, input=self.input)

        x = self.input
        k = (2 / pi) ** 0.5

        u = k * (x + 0.044715 * x**3)
        tanh_term = u.tanh()
        sech2_term = 1 - tanh_term**2

        gelu_grad = 0.5 * (1 + tanh_term) + 0.5 * x * sech2_term * k * (
            1 + 0.134145 * x**2
        )

        return grad * gelu_grad


class Sigmoid(Activation):
    """Sigmoid activation layer."""

    def __init__(self) -> None:
        super().__init__("Sigmoid")

    def forward(self, x: Any) -> Any:
        """Apply Sigmoid activation element-wise.

        Args:
            x: Input :class:`~structs.Array`.

        Returns:
            ``1 / (1 + exp(-x))``; the result is cached for ``backward``.
        """
        self.input = x
        logger.debug("sigmoid forward", value=x)
        out = 1 / (1 + -x.exp())
        self.output = out
        return out if isinstance(out, Array) else Array([out])

    def backward(self, grad: Any) -> Any:
        """Compute backward pass for Sigmoid activation.

        Args:
            grad: Upstream gradient.

        Returns:
            ``grad * output * (1 - output)`` using the cached forward output.
        """
        logger.debug("sigmoid backward", grad=grad, input=self.input)
        out = self.output
        return grad * (out * (1 - out))


class Tanh(Activation):
    """Tanh activation layer."""

    def __init__(self) -> None:
        super().__init__("Tanh")

    def forward(self, x: Any) -> Any:
        """Apply Tanh activation element-wise.

        Computed from exponentials: ``(e^x - e^-x) / (e^x + e^-x)``.

        Args:
            x: Input :class:`~structs.Array`.

        Returns:
            Tanh output; the result is cached for ``backward``.
        """
        self.input = x
        logger.debug("tanh forward", value=x)
        e_px, e_nx = x.exp(), (-x).exp()
        out = (e_px - e_nx) / (e_px + e_nx)
        self.output = out
        return out if isinstance(out, Array) else Array([out])

    def backward(self, grad: Any) -> Any:
        """Compute backward pass for Tanh activation.

        Args:
            grad: Upstream gradient.

        Returns:
            ``grad * (1 - output^2)`` using the cached forward output.
        """
        logger.debug("tanh backward", grad=grad, input=self.input)
        return grad * (1 - (self.output**2))


class Softmax(Activation):
    """Numerically stable softmax activation.

    Applies softmax along *axis* (default: last axis).
    Input / output shapes:
      * 1-D (C,)    → (C,)
      * 2-D (N, C)  → (N, C)  applied row-wise
    """

    def __init__(self) -> None:
        """Softmax activation layer."""
        super().__init__("Softmax")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_1d(row: Array) -> Array:
        """Numerically stable softmax on a 1-D ``Array``.

        Args:
            row: 1-D logits array.

        Returns:
            Probabilities summing to one along the row.
        """
        m = row.max()  # subtract max to prevent exp overflow
        e = (row - m).exp()
        return e / e.sum()

    @staticmethod
    def _grad_1d(p: Array, g: Array) -> Array:
        """Jacobian–vector product for a single softmax output.

        dx_i = p_i * (g_i - sum_j(g_j * p_j))

        Args:
            p: Softmax probabilities (forward output).
            g: Upstream gradient for this row.

        Returns:
            Gradient w.r.t. the input logits of this row.
        """
        return p * (g - (p * g).sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x: Union[Array, Any], axis: int = -1) -> Array:
        """Apply softmax along *axis*.

        Parameters
        ----------
        x : Array
            Logits of shape (C,) or (N, C).
        axis : int
            Axis along which softmax is computed (default -1, i.e. last).

        Returns
        -------
        Array  same shape as x.

        Raises:
            ShapeError: If ``x`` is neither 1-D nor 2-D.
        """
        if not isinstance(x, Array):
            x = Array(x)
        self.input = x
        logger.debug("softmax forward", activation=self.name, shape=x.shape)

        if x.ndim == 1:
            out = self._softmax_1d(x)
        elif x.ndim == 2:
            # apply row-wise along axis=1 (last axis)
            out = Array.stack(
                [self._softmax_1d(row) for row in x.unbind(axis=0)],
                axis=0,
            )
        else:
            raise ShapeError(
                f"Softmax supports 1-D and 2-D inputs, got shape {x.shape}"
            )

        self.output = out
        logger.debug("softmax forward done", activation=self.name, shape=out.shape)
        return out

    def backward(self, grad: Union[Array, Any]) -> Array:
        """Compute gradient of softmax w.r.t. its input.

        Uses the compact Jacobian–vector product form instead of the full
        C×C Jacobian: dx = p * (g - dot(g, p))

        Parameters
        ----------
        grad : Array
            Upstream gradient, same shape as forward output.

        Returns
        -------
        Array  same shape as grad.
        """
        if not isinstance(grad, Array):
            grad = Array(grad)
        p = self.output
        logger.debug("softmax backward", activation=self.name)

        if p.ndim == 1:
            return self._grad_1d(p, grad)

        return Array.stack(
            [
                self._grad_1d(pi, gi)
                for pi, gi in zip(p.unbind(axis=0), grad.unbind(axis=0))
            ],
            axis=0,
        )
