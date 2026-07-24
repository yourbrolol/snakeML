from basic.layers import Layer
from debug import get_logger
from structs import Array

logger = get_logger(__name__)

class Activation(Layer):
    def __init__(self, name):
        """Base activation layer initializing name and input state."""
        super().__init__()
        self.name = name
        self.input = None
    def forward(self, x): raise NotImplementedError
    def backward(self, x): raise NotImplementedError

class LinearActivation(Activation):
    def __init__(self):
        """Identity activation layer."""
        super().__init__("LinearActivation")
    def forward(self, x):
        """Pass input through without transformation."""
        logger.debug("linear activation forward", activation=self.name, value=x)
        return x
    def backward(self, x):
        """Pass gradient through without transformation."""
        logger.debug("linear activation backward", activation=self.name, value=x)
        return x

class ReLU(Activation):
    def __init__(self):
        """Rectified Linear Unit activation layer."""
        super().__init__("ReLU")
    def forward(self, x):
        """Apply ReLU activation element-wise."""
        self.input = x
        logger.debug("relu forward", value=x)
        out = max(0, x)
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        """Compute backward pass for ReLU activation."""
        logger.debug("relu backward", grad=grad, input=self.input)
        return grad * (self.input > 0)

class Softmax(Activation):
    def __init__(self):
        """Softmax activation layer."""
        super().__init__("Softmax")
    def forward(self, x, axis=-1):
        pass