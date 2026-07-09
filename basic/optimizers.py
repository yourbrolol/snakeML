class Optimizer():
    def __init__(self, layers=[], lr=0.01):
        self.layers = layers
        self.lr = 0.01
    def step(self): raise NotImplementedError
    def zero_grad(self): raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, layers=[], lr=0.01):
        super().__init__(layers, lr)
    def step(self):
        for layer in self.layers:
            if not layer.params or not layer.grads: continue
            layer.params['w'] -= self.lr * layer.grads['w']
            layer.params['b'] -= self.lr * layer.grads['b']
    def zero_grad(self):
        for layer in self.layers:
            if not layer.grads: continue
            layer.grads = {}
