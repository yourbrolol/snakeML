from structs import Array

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
        # print("Input: ", self.input, self.params['w'], self.params['b'], self.params['w'].ndim, self.input.ndim)
        print(self.params['w'].shape, self.input.shape)
        out = (self.params['w'].matmul(self.input)) + self.params['b']
        # print("Output: ", out)
        return out
    def update(self):
        self.params['w'] -= self.grads['w']
        self.params['b'] -= self.grads['b']
    def backward(self, loss_grad):
        grad = Array(loss_grad) if not isinstance(loss_grad, Array) else loss_grad
        self.grads['w'] = grad.outer(self.input)
        self.grads['b'] = grad
        # print(self.input, grad, self.grads, self.params)
        return self.params['w'].T.matmul(grad)
