class Activation:
    def __init__(self, name, fn, grad):
        self.name = name
        self.fn = fn
        self.grad = grad

class LinearActivation(Activation):
    def __init__(self): super().__init__("LinearActivation", lambda x: x, lambda x: x)

class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU", lambda x: x if x>0 else 0, lambda x: 0 if x==0 else 1)
