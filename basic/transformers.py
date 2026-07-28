from .layers import Layer
from structs import Array
from basic.optimizers import SGD

class Embedding(Layer):
    def __init__(self, vocab_size, d_model, table=None):
        super().__init__()
        if table is not None:
            self.params['w'] = Array.wraparray(table)
        else:
            self.params['w'] = Array.randn((vocab_size, d_model))
        self.grads['w'] = Array.zeros((vocab_size, d_model))
    def forward(self, x):
        self.input = x
        return self.params['w'][x]
    def backward(self, grad):
        for token, g in zip(self.input, grad):
            self.grads["w"][token] += g
        return None

class LayerNorm(Layer):
    def __init__(self, input_size):
        super().__init__()
        self.params['w'] = Array.ones((input_size,))
        self.params['b'] = Array.zeros((input_size,))
    def forward(self, input, eps=1e-5):
        self.input = Array.wraparray(input)
        x = self.input
        self.eps = eps
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = ((x - self.mean) ** 2).mean(axis=-1, keepdims=True)
        self.inv_std = 1 / (self.var + eps).sqrt()
        self.x_hat = (x - self.mean) * self.inv_std
        return self.params['w'] * self.x_hat + self.params['b']
    def backward(self, grad):
        # parameter gradients
        self.grads['w'] = (grad * self.x_hat).sum(axis=tuple(range(grad.ndim - 1)))
        self.grads['b'] = grad.sum(axis=tuple(range(grad.ndim - 1)))
        # input gradient
        dx_hat = grad * self.params['w']
        mean1 = dx_hat.mean(axis=-1, keepdims=True)
        mean2 = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        dX = self.inv_std * (dx_hat - mean1 - self.x_hat * mean2)
        return dX

if __name__ == "__main__":
    norm = LayerNorm(3)
    opt = SGD([norm], lr=0.01)
    x = [1.0, 2.0, 3.0]
    print(norm.forward(x))
    norm.backward(Array([0.1, 0.2, 0.3]))
    opt.step()
    print(norm.params['w'], norm.params['b'])