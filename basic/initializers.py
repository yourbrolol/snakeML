from math import sqrt
from structs import Array
from structs.utils import newarr
from matlib.rng import normal

class Initializer:
    def __init__(self):
        self.params = {}
    def __call__(self, shape): raise NotImplementedError

class XavierNormal(Initializer):
    def __init__(self): super().__init__()
    def __call__(self, shape, fan_in, fan_out):
        sigma = sqrt(2/(fan_in+fan_out))
        return Array(newarr(shape, lambda: normal(0, sigma)))

if __name__ == "__main__":
    print(XavierNormal()([3,3], 12, 12))
