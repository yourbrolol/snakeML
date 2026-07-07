from basic.layers import Layer

class Activation(Layer):
    def __init__(self, name):
        super().__init__()
        self.name = name
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
    def forward(self, x): return max(0, x)
    def backward(self, x): return 0 if x <= 0 else 1
