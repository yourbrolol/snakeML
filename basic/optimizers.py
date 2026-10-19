"""Gradient-based parameter optimizers.

:class:`Optimizer` owns a list of layers and a learning rate, and walks
possibly-nested parameter/gradient dicts leaf-by-leaf. Subclasses
(:class:`SGD`, :class:`Adam`) implement ``step``; all support
``zero_grad``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from debug import get_logger
from structs import Array

logger = get_logger(__name__)

__all__ = ["Adam", "Optimizer", "SGD"]


class Optimizer:
    """Base optimizer class storing layers and learning rate.

    Attributes:
        layers: Ordered layers whose ``params``/``grads`` are updated.
        lr: Learning rate scaling every update.
    """

    def __init__(self, layers: Optional[Sequence[Any]] = None, lr: float = 0.01) -> None:
        self.layers: Optional[Sequence[Any]] = [] if layers is None else layers
        self.lr = lr

    def _recursive_items(
        self,
        params: Dict[str, Any],
        grads: Dict[str, Any],
        prefix: str = "",
    ) -> Iterator[Tuple[str, Dict[str, Any], str, Any]]:
        """Yield ``(path, params_dict, key, grad)`` for every leaf parameter with a gradient.

        Args:
            params: Possibly-nested parameter mapping.
            grads: Gradient mapping mirroring ``params`` (may be sparser).
            prefix: Dotted key-path accumulated during the descent.

        Yields:
            A 4-tuple per leaf: the dotted ``path``, the owning ``params``
            dict, its ``key`` within that dict, and the matching ``grad``.
        """
        for key in params:
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(params[key], dict):
                yield from self._recursive_items(params[key], grads.get(key, {}), path)
            elif key in grads:
                yield path, params, key, grads[key]

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, layers: Optional[Sequence[Any]] = None, lr: float = 0.01) -> None:
        super().__init__(layers, lr)

    def step(self) -> None:
        """Perform a single optimization step updating layer parameters.

        Skips layers lacking params or gradients; each parameter is
        updated in-place as ``param -= lr * grad``.
        """
        logger.debug("optimizer step", layer_count=len(self.layers), lr=self.lr)
        for layer in self.layers:
            if not layer.params or not layer.grads:
                logger.debug(
                    "optimizer skipping layer with no params or grads", layer=layer
                )
                continue
            for _, params, key, grad in self._recursive_items(
                layer.params, layer.grads
            ):
                params[key] -= self.lr * grad

    def zero_grad(self) -> None:
        """Clear gradients of all managed layers."""
        logger.debug("optimizer zero_grad", layer_count=len(self.layers))

        for layer in self.layers:
            layer.grads = {}


class Adam(Optimizer):
    """Adam optimizer with bias-corrected first/second moment estimates.

    Attributes:
        beta1: Exponential decay rate for the first moment (m).
        beta2: Exponential decay rate for the second moment (v).
        epsilon: Numerical stabilizer in the update denominator.
        m: First moment estimates keyed by dotted parameter path.
        v: Second moment estimates keyed by dotted parameter path.
        t: Number of ``step`` calls performed (for bias correction).
    """

    def __init__(
        self,
        layers: Optional[Sequence[Any]] = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(layers, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m: Dict[str, Any] = {}
        self.v: Dict[str, Any] = {}
        self.t: int = 0

    def step(self) -> None:
        """Perform a single optimization step updating layer parameters.

        Increments :attr:`t`, lazily initializes moment arrays per path,
        updates the EMA moments ``m``/``v`` and applies the
        bias-corrected Adam update ``param -= lr * m_hat / (sqrt(v_hat) + eps)``.
        """
        logger.debug("optimizer step", layer_count=len(self.layers), lr=self.lr)

        self.t += 1

        for layer in self.layers:
            if not layer.params or not layer.grads:
                continue

            for path, params, key, grad in self._recursive_items(
                layer.params, layer.grads
            ):
                if path not in self.m:
                    self.m[path] = Array.zeros(grad.shape)
                    self.v[path] = Array.zeros(grad.shape)

                self.m[path] = self.beta1 * self.m[path] + (1 - self.beta1) * grad

                self.v[path] = self.beta2 * self.v[path] + (1 - self.beta2) * (grad**2)

                m_hat = self.m[path] / (1 - self.beta1**self.t)
                v_hat = self.v[path] / (1 - self.beta2**self.t)

                params[key] -= self.lr * m_hat / (v_hat**0.5 + self.epsilon)

    def zero_grad(self) -> None:
        """Clear gradients of all managed layers."""
        logger.debug("optimizer zero_grad", layer_count=len(self.layers))

        for layer in self.layers:
            layer.grads = {}
