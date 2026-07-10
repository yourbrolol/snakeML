from basic.layers import Layer
from structs import Array

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
    def forward(self, x): return x
    def backward(self, x): return x

class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    def forward(self, x):
        self.input = x
        out = max(0, x)
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        return grad * (self.input > 0)
