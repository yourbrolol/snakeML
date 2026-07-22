from basic.layers import Layer
from debug import get_logger
from structs import Array

logger = get_logger(__name__)

class Activation(Layer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.input = None
    def forward(self, x): raise NotImplementedError
    def backward(self, x): raise NotImplementedError

class LinearActivation(Activation):
    def __init__(self):
        super().__init__("LinearActivation")
    def forward(self, x):
        logger.debug("linear activation forward", activation=self.name, value=x)
        return x
    def backward(self, x):
        logger.debug("linear activation backward", activation=self.name, value=x)
        return x

class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    def forward(self, x):
        self.input = x
        logger.debug("relu forward", value=x)
        out = max(0, x)
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        logger.debug("relu backward", grad=grad, input=self.input)
        return grad * (self.input > 0)
