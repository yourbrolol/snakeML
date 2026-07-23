from math import sqrt
from structs import Array
from structs.utils import newarr
from matlib.rng import normal

class Initializer:
    def __init__(self):
        """Base parameter initializer class."""
        self.params = {}
    def __call__(self, shape): raise NotImplementedError

class Constant(Initializer):
    def __init__(self):
        """Constant parameter initializer (zeros)."""
        super().__init__()
    def __call__(self, shape):
        """Return an array of the given shape filled with constant zeros."""
        return Array(newarr(shape, lambda: 0))

class XavierNormal(Initializer):
    def __init__(self):
        """Xavier/Glorot normal parameter initializer."""
        super().__init__()
    def __call__(self, shape, fan_in, fan_out):
        """Return an array sampled from a zero-mean normal distribution with variance 2/(fan_in + fan_out)."""
        sigma = sqrt(2/(fan_in+fan_out))
        return Array(newarr(shape, lambda: normal(0, sigma)))

if __name__ == "__main__":
    print(XavierNormal()([3,3], 12, 12))
