"""Model containers composing layers into pipelines.

:class:`Sequential` chains layers forward in order and gradients
backward in reverse; :class:`LayerList` is a ``list`` subclass that also
exposes its members via attribute access and index-path parameter keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from basic.layers import Layer
from debug import get_logger, operation_context

logger = get_logger(__name__)

__all__ = ["LayerList", "Sequential"]


class LayerList(list):
    """A list-like container for layers that allows attribute access to submodules.

    Attributes:
        submodules: Mapping of index-string keys (``"0"``, ``"1"``...)
            to the contained layers.
    """

    def __init__(self, *args: Sequence[Layer]) -> None:
        super().__init__(*args)
        self.submodules = {f"{i}": layer for i, layer in enumerate(self)}

    def __getattr__(self, name: str) -> Layer:
        """Return layer ``name`` from ``submodules``.

        Args:
            name: Attribute (index-string key) to look up.

        Returns:
            The contained layer registered under ``name``.

        Raises:
            AttributeError: If ``name`` is not a registered submodule.
        """
        if name in self.submodules:
            return self.submodules[name]
        raise AttributeError(f"'LayerList' object has no attribute '{name}'")

    def parameters(self) -> Dict[str, Layer]:
        """Recursively collect parameters from all layers in the list.

        Returns:
            A flat mapping with ``"<index>.<param>"`` keys.
        """
        params = {}
        for i, layer in enumerate(self):
            params.update({f"{i}.{k}": v for k, v in layer.parameters().items()})
        return params


class Sequential:
    """A forward/backward pipeline over an ordered list of layers.

    Attributes:
        layers: The wrapped layers (may be ``None`` until populated).
        grads: Collected gradients from the most recent backward pass.
        input: Cached input of the most recent forward pass.
    """

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        """Initialize a sequential container with an ordered list of layers."""
        self.layers = layers
        self.grads: Dict[str, Any] = {}
        self.input: Optional[Any] = None

    def forward(self, input_data: Any) -> Any:
        """Pass input sequentially through all layers in order.

        Args:
            input_data: Input fed into the first layer.

        Returns:
            Output of the final layer, or ``input_data`` unchanged when no
            layers are present.

        Raises:
            TypeError: If ``layers`` is ``None`` (not iterable).
        """
        with operation_context("model.forward", layer_count=len(self.layers)):
            out = input_data
            for layer in self.layers:
                out = layer.forward(out)
            logger.debug("sequence forward complete", step=len(self.layers))
            return out

    def backward(self, loss_grad: Any) -> Any:
        """Pass loss gradient backward through all layers in reverse order.

        Args:
            loss_grad: Gradient at the output of the final layer.

        Returns:
            Gradient at the original input after the reverse chain.

        Raises:
            TypeError: If ``layers`` is ``None`` (not iterable).
        """
        with operation_context("model.backward", layer_count=len(self.layers)):
            grad = loss_grad
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            logger.debug("sequence backward complete", step=len(self.layers))
            return grad
