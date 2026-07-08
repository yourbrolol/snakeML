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
        self.params['w'] = Array([0.1] * input_dim)
        self.params['b'] = Array([0.1])
    def forward(self, input_data):
        self.input = Array(input_data) if not isinstance(input_data, Array) else input_data
        print("cur", self.input, self.params['w'], self.params['b'])
        print("res", ((self.input @ self.params['w']) + self.params['b']))
        return (self.input @ self.params['w']) + self.params['b']
    def backward(self, output_grad):
        self.params['w'] -= output_grad * self.input
        self.params['b'] -= output_grad
