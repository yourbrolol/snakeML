from basic.structs import Array

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
        self.params['w'] = Array([[0.1] * input_dim for _ in range(output_dim)])
        self.params['b'] = Array([0.1] * output_dim)
    def forward(self, input_data):
        self.input = Array(input_data) if not isinstance(input_data, Array) else input_data
        print("cur", self.input, self.params['w'], self.params['b'], self.params['w'].ndim, self.input.ndim)
        print("res", ((self.params['w'].dot(self.input)) + self.params['b']))
        return (self.params['w'].dot(self.input)) + self.params['b']
    def backward(self, output_grad):
        self.params['w'] -= self.input.outer(output_grad)
        self.params['b'] -= output_grad
