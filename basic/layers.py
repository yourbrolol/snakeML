from structs import Array
from basic.utils import pool2d
from math import isqrt

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
        print(self.params['w'].shape, self.input.shape)
        out = (self.params['w'].matmul(self.input)) + self.params['b']
        return out
    def backward(self, loss_grad):
        grad = Array(loss_grad) if not isinstance(loss_grad, Array) else loss_grad
        self.grads['w'] = grad.outer(self.input)
        self.grads['b'] = grad
        return self.params['w'].T.matmul(grad)

class Conv2D(Layer):
    def __init__(self, input_dim, output_dim, kernel, stride=[1,1]):
        super().__init__()
        self.params['w'] = Array([[[[0.1 for _ in range(kernel[1])] for _ in range(kernel[0])] for _ in range(input_dim)] for _ in range(output_dim)])
        self.params['b'] = Array([0.1] * output_dim)
        self.params['strd'] = stride
        self.params['krnl'] = kernel
        self.input = None
    def forward(self, input):
        self.input = Array.wraparray(input)
        out = (self.params['w'].dot(Array.wraparray((pool2d(self.input, self.params['krnl'], self.params['strd']))), axes=[[1,2,3], [1,2,3]]) + self.params['b']).reshape(3,1,3,3)
        return out
    def backward(self, grad):
        self.grads['w'], self.grads['b'], dX = grad.dot(self.input, [[1], [0]]), grad, grad.dot(self.params['w'], [[0], [0]])
        return dX

class MaxPool2D(Layer):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    model = Conv2D(1, 3, [2, 2], [1,1])
    x = Array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]])
    print(model.forward(x))
