from .layers import Layer
from structs import Array

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