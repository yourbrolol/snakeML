from layers import Layer

class Sequential():
    def __init__(self, layers: list[Layer] = []):
        self.layers = layers
        self.grads = {}
        self.input = None
    def forward(self, input_data):
        out = input_data
        for layer in self.layers: out = layer.forward(out)
        return out
