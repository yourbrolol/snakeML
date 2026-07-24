from basic.layers import Layer
from debug import get_logger, operation_context

logger = get_logger(__name__)

class LayerList(list):
    """A list-like container for layers that allows attribute access to submodules."""
    def __init__(self, *args):
        super().__init__(*args)
        self.submodules = {f"{i}": layer for i, layer in enumerate(self)}
    def __getattr__(self, name):
        if name in self.submodules:
            return self.submodules[name]
        raise AttributeError(f"'LayerList' object has no attribute '{name}'")
    def parameters(self):
        """Recursively collect parameters from all layers in the list."""
        params = {}
        for i, layer in enumerate(self):
            params.update({f"{i}.{k}": v for k, v in layer.parameters().items()})
        return params

class Sequential():
    def __init__(self, layers: list[Layer] | None = None):
        """Initialize a sequential container with an ordered list of layers."""
        self.layers = layers
        self.grads = {}
        self.input = None
    def forward(self, input_data):
        """Pass input sequentially through all layers in order."""
        with operation_context("model.forward", layer_count=len(self.layers)):
            out = input_data
            for layer in self.layers:
                out = layer.forward(out)
            logger.debug("sequence forward complete", step=len(self.layers))
            return out
    def backward(self, loss_grad):
        """Pass loss gradient backward through all layers in reverse order."""
        with operation_context("model.backward", layer_count=len(self.layers)):
            grad = loss_grad
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            logger.debug("sequence backward complete", step=len(self.layers))
            return grad
