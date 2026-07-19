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
    def update(self):
        self.params['w'] -= self.grads['w']
        self.params['b'] -= self.grads['b']
    def backward(self, loss_grad):
        grad = Array(loss_grad) if not isinstance(loss_grad, Array) else loss_grad
        self.grads['w'] = grad.outer(self.input)
        self.grads['b'] = grad
        return self.params['w'].T.matmul(grad)

class Conv2D(Layer):
    def __init__(self, input_dim, output_dim, kernel, stride=[1,1]):
        super().__init__()
        self.params['w'] = Array([[0.1] * kernel[1]] * kernel[0] for _ in range(output_dim))
        self.params['b'] = Array([0.1] * output_dim)
        self.params['strd'] = stride
        self.params['krnl'] = kernel
    def forward(self, input):
        out = []
        for kw, kb in zip(self.params['w'], self.params['b']):
            pooled = pool2d(input, self.params['krnl'], self.params['strd'])
            n = len(pooled)
            dim = isqrt(n)
            res = [i.dot(kw, axes=2)+kb for i in iter(pooled)]
            decomposed = [res[i : i + dim] for i in range(0, n, dim)]
            out.append(decomposed)
        return Array(out)

class MaxPool2D(Layer):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    model = Conv2D(10, 3, [2, 2], [1,1])
    x = Array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    print(model.forward(x))
