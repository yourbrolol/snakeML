from .structs import Vector

class Neuron():
    def __init__(self, input_dim):
        self.w = Vector()
        self.bias = Vector()
        self.input = None
        self.d_w = None
        self.d_b = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_grad): raise NotImplementedError

class LinearNeuron(Neuron):
    def __init__(self, input_dim):
        self.w = Vector([0.1] * input_dim)
        self.bias = Vector(0.1)
        self.input = None
        self.d_w = None
        self.d_b = None
    def forward(self, input_data):
        self,input = input_data
        return input_data * self.w + self.b
    def backward(self, output_grad):
        self.d_w = output_grad / self.input
        self.d_b = output_grad
        return output_grad * self.w

class NeuronBasedLinear:
    def __init__(self, input_dim, output_dim):
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]

    def forward(self, input_data):
        return Vector([neuron.forward(input_data) for neuron in self.neurons])

    def backward(self, output_gradient):
        input_gradient = Vector([0] * len(self.neurons[0].w))
        
        for i, neuron in enumerate(self.neurons):
            input_gradient += neuron.backward(output_gradient[i])
            
        return input_gradient

class Layer():
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input = None
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_gradient): raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.params['w'] = Vector([0.1] * input_dim)
        self.params['b'] = Vector(0.1)
    def forward(self, input_data):
        self.input = input_data
        return self.input @ self.params['w'] + self.params['b']
