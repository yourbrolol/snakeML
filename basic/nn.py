from basic.layers import Layer
from debug import get_logger, operation_context

logger = get_logger(__name__)

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
