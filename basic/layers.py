from basic.structs import Vector

class Layer():
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_grad): raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.params['w'] = Vector([0.1] * input_dim)
        self.params['b'] = Vector([0.1])
    def forward(self, input_data):
        self.input = Vector(input_data)
        return self.input @ self.params['w'] + self.params['b']
    def backward(self, output_grad):
        self.params['w'] -= output_grad * self.input
        self.params['b'] -= output_grad
