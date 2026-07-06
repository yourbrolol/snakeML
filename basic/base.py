from .structs import Vector

class Layer():
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_gradient): raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.params['w'] = Vector([0.1] * input_dim)
        self.params['b'] = Vector(0.1)
    def forward(self, input_data):
        self.input = input_data
        return self.input @ self.params['w'] + self.params['b']
