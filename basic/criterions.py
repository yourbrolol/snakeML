from basic.structs import Array

class Loss:
    def __init__(self, name):
        self.name = name
    def forward(self, y_pred, y_true): raise NotImplementedError
    def backward(self, y_pred, y_true): raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        super().__init__("MSELoss")
    def forward(self, y_pred, y_true): return (1/2) * (Array([y_pred])-Array([y_true]))**2
    def backward(self, y_pred, y_true): return (Array([y_pred]) - Array([y_true]))
