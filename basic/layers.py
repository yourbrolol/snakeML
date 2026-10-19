"""Neural network layers: the core differentiable building blocks.

:class:`Layer` defines the shared parameter/submodule plumbing; concrete
subclasses (:class:`Linear`, :class:`Conv2D`, :class:`MaxPool2D`)
implement ``forward``/``backward`` and populate ``grads`` for the
optimizers in :mod:`basic.optimizers`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from basic.initializers import Constant
from basic.utils import im2col
from debug import get_logger, operation_context
from structs import Array

logger = get_logger(__name__)

__all__ = ["Conv2D", "Layer", "Linear", "MaxPool2D"]


class Layer:
    """Base class for all neural network layers.

    Attributes:
        params: Trainable parameters, keyed by name.
        grads: Gradients matching ``params`` layout, populated by ``backward``.
        submodules: Child layers whose ``parameters()`` are collected per
            key path (``"name.param"``).
        input: Cached forward-pass input used by ``backward``.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}
        self.grads: Dict[str, Any] = {}
        self.submodules: Dict[str, "Layer"] = {}
        self.input: Optional[Any] = None

    def forward(self, input_data: Any) -> Any:
        raise NotImplementedError

    def backward(self, output_grad: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> Dict[str, Any]:
        """Recursively collect all parameters from this layer and child submodules.

        Returns:
            A flat mapping of ``"<scope>.<name>"`` keys to parameter arrays.
        """
        params = dict(self.params)
        for name, module in self.submodules.items():
            params.update({f"{name}.{k}": v for k, v in module.parameters().items()})
        return params


class Linear(Layer):
    """Fully-connected (affine) layer: ``out = w @ input + b``.

    Attributes:
        params["w"]: Weight matrix of shape ``(output_dim, input_dim)``.
        params["b"]: Output-dimension bias vector.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: Optional[Any] = None,
    ) -> None:
        """Initialize weights via ``initializer`` (Xavier normal by default) and zero bias."""
        super().__init__()
        if initializer is None:
            from basic.initializers import XavierNormal

            initializer = XavierNormal()
        self.params["w"] = initializer([output_dim, input_dim], input_dim, output_dim)
        self.params["b"] = Constant()(0, [output_dim])

    def forward(self, input_data: Any) -> Array:
        """Compute the forward pass for the linear transformation.

        Args:
            input_data: Input vector/matrix; lists are coerced to ``Array``.

        Returns:
            ``w.matmul(input) + b`` with shapes broadcast per ``b``.
        """
        self.input = (
            Array(input_data) if not isinstance(input_data, Array) else input_data
        )
        with operation_context(
            "linear.forward",
            input_shape=self.input.shape,
            weight_shape=self.params["w"].shape,
        ):
            logger.debug(
                "linear forward %s -> %s", self.params["w"].shape, self.input.shape
            )
            out = (self.params["w"].matmul(self.input)) + self.params["b"]
            return out

    def backward(self, loss_grad: Any) -> Array:
        """Compute gradients with respect to weights, bias, and input.

        Populates ``grads["w"]`` (outer product with the cached input) and
        ``grads["b"]`` (the raw loss gradient).

        Args:
            loss_grad: Upstream gradient of this layer's output.

        Returns:
            Gradient w.r.t. the layer input, ``w.T.matmul(grad)``.
        """
        grad = Array(loss_grad) if not isinstance(loss_grad, Array) else loss_grad
        self.grads["w"] = grad.outer(self.input)
        self.grads["b"] = grad
        logger.debug("linear backward grad=%s", grad.shape, operation="linear.backward")
        return self.params["w"].T.matmul(grad)


class Conv2D(Layer):
    """2D convolution implemented via :func:`basic.utils.im2col` + tensordot.

    Input is ``(C, H, W)``; output is ``(out_c, 1, out_h, out_w)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel: Sequence[int],
        stride: Sequence[int] = [1, 1],
        initializer: Optional[Any] = None,
    ) -> None:
        """Initialize 2D convolutional layer with weights, bias, kernel, and stride.

        Args:
            input_dim: Number of input channels.
            output_dim: Number of output channels (filters).
            kernel: Two-element ``[KH, KW]`` kernel size.
            stride: Two-element ``[SH, SW]`` stride.
            initializer: Weight initializer; falls back to Xavier normal.
        """
        super().__init__()
        if initializer is None:
            from basic.initializers import XavierNormal

            initializer = XavierNormal()
        self.params["w"] = initializer(
            [output_dim, input_dim, *kernel], input_dim, output_dim
        )
        self.params["b"] = Constant()(0, [output_dim])
        self.params["strd"] = stride
        self.params["krnl"] = kernel
        self.input = None

    def forward(self, input: Any) -> Array:
        """Compute 2D convolution forward pass over input patches.

        Args:
            input: Input ``Array`` (or list) of shape ``(C, H, W)``.

        Returns:
            Feature map of shape ``(out_c, 1, out_h, out_w)`` where
            ``out_h = (H - KH) // SH + 1`` and ``out_w`` analogous.
        """
        self.input = Array.wraparray(input)
        with operation_context(
            "conv2d.forward",
            input_shape=self.input.shape,
            kernel=self.params["krnl"],
            stride=self.params["strd"],
        ):
            logger.debug(
                "conv2d forward",
                input_shape=self.input.shape,
                kernel=self.params["krnl"],
                stride=self.params["strd"],
            )
            KH, KW = self.params["krnl"]
            SH, SW = self.params["strd"]
            out_h = (self.input.shape[1] - KH) // SH + 1
            out_w = (self.input.shape[2] - KW) // SW + 1
            out_c = self.params["w"].shape[0]
            out = (
                self.params["w"].dot(
                    Array.wraparray(
                        (im2col(self.input, self.params["krnl"], self.params["strd"]))
                    ),
                    axes=[[1, 2, 3], [1, 2, 3]],
                )
                + self.params["b"]
            ).reshape(out_c, 1, out_h, out_w)
            logger.debug("conv2d output", output_shape=out.shape)
            return out

    def backward(self, grad: Any) -> Array:
        """Compute gradients for Conv2D layer parameters and input.

        Args:
            grad: Upstream gradient of shape ``(out_c, 1, out_h, out_w)``.

        Returns:
            Gradient w.r.t. the input, via tensordot with the kernel.
        """
        self.grads["w"], self.grads["b"], dX = (
            grad.dot(self.input, [[1], [0]]),
            grad,
            grad.dot(self.params["w"], [[0], [0]]),
        )
        logger.debug(
            "conv2d backward",
            grad_shape=getattr(grad, "shape", None),
            input_shape=getattr(self.input, "shape", None),
        )
        return dX


class MaxPool2D(Layer):
    """Placeholder max-pooling layer; pooling is not yet implemented."""


if __name__ == "__main__":
    model = Conv2D(1, 3, [2, 2], [1, 1])
    x = Array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    logger.info("forward result %s", model.forward(x), operation="main")
