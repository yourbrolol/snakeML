from basic.structs import Array

class Loss:
    def __init__(self, name):
        self.name = name
    def forward(self, y_pred, y_true): raise NotImplementedError
    def backward(self, y_pred, y_true): raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        super().__init__("MSELoss")
    def forward(self, y_pred, y_true): return (1/2) * ((y_pred)-(Array(y_true) if not isinstance(y_true, Array) else y_true))**2
    def backward(self, y_pred, y_true): return (y_pred) - (Array(y_true) if not isinstance(y_true, Array) else y_true) 
