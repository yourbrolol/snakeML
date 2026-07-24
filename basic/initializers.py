import random
from math import sqrt

from matlib.rng import normal
from structs import Array
from structs.utils import newarr


class Initializer:
    def __init__(self):
        """Base parameter initializer class."""
        self.params = {}

    def __call__(self, shape):
        raise NotImplementedError


class Constant(Initializer):
    def __init__(self):
        """Constant parameter initializer (zeros)."""
        super().__init__()

    def __call__(self, shape):
        """Return an array of the given shape filled with constant zeros."""
        return Array(newarr(shape, lambda: 0))


class Zeros(Initializer):
    def __init__(self):
        """Zero-valued parameter initializer."""
        super().__init__()

    def __call__(self, shape):
        """Return an array of the given shape filled with zeros."""
        return Array(newarr(shape, lambda: 0))


class Ones(Initializer):
    def __init__(self):
        """One-valued parameter initializer."""
        super().__init__()

    def __call__(self, shape):
        """Return an array of the given shape filled with ones."""
        return Array(newarr(shape, lambda: 1))


class Normal(Initializer):
    def __init__(self):
        """Random normal initializer."""
        super().__init__()

    def __call__(self, shape, mean=0.0, std=1.0):
        """Return an array sampled from a normal distribution."""
        return Array(newarr(shape, lambda: normal(mean, std)))


class Uniform(Initializer):
    def __init__(self):
        """Random uniform initializer."""
        super().__init__()

    def __call__(self, shape, low=0.0, high=1.0):
        """Return an array sampled uniformly from [low, high)."""
        return Array(newarr(shape, lambda: random.uniform(low, high)))


class XavierNormal(Initializer):
    def __init__(self):
        """Xavier/Glorot normal parameter initializer."""
        super().__init__()

    def __call__(self, shape, fan_in, fan_out):
        """Return an array sampled from a normal distribution with variance 2/(fan_in + fan_out)."""
        sigma = sqrt(2 / (fan_in + fan_out))
        return Array(newarr(shape, lambda: normal(0, sigma)))


class XavierUniform(Initializer):
    def __init__(self):
        """Xavier/Glorot uniform parameter initializer."""
        super().__init__()

    def __call__(self, shape, fan_in, fan_out):
        """Return an array sampled uniformly from [-limit, limit]."""
        limit = sqrt(6 / (fan_in + fan_out))
        return Array(newarr(shape, lambda: random.uniform(-limit, limit)))


class KaimingNormal(Initializer):
    def __init__(self):
        """Kaiming normal parameter initializer."""
        super().__init__()

    def __call__(self, shape, fan_in):
        """Return an array sampled from a normal distribution with std sqrt(2/fan_in)."""
        sigma = sqrt(2 / fan_in)
        return Array(newarr(shape, lambda: normal(0, sigma)))


class KaimingUniform(Initializer):
    def __init__(self):
        """Kaiming uniform parameter initializer."""
        super().__init__()

    def __call__(self, shape, fan_in):
        """Return an array sampled uniformly from [-limit, limit]."""
        limit = sqrt(6 / fan_in)
        return Array(newarr(shape, lambda: random.uniform(-limit, limit)))


if __name__ == "__main__":
    print(XavierNormal()([3, 3], 12, 12))
