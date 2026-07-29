from debug import get_logger
from structs import Array

logger = get_logger(__name__)

def _recursive_items(self, params, grads, prefix=""):
    for key in params:
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(params[key], dict):
            yield from self._recursive_items(
                params[key],
                grads.get(key, {}),
                path
            )
        else:
            if key in grads:
                yield path, params, key, grads[key]

class Optimizer():
    def __init__(self, layers=None, lr=0.01):
        """Base optimizer class storing layers and learning rate."""
        self.layers = [] if layers is None else layers
        self.lr = lr
    def step(self): raise NotImplementedError
    def zero_grad(self): raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, layers=None, lr=0.01):
        """Stochastic Gradient Descent optimizer."""
        super().__init__(layers, lr)
    def step(self):
        """Perform a single optimization step updating layer parameters."""
        logger.debug("optimizer step", layer_count=len(self.layers), lr=self.lr)
        for layer in self.layers:
            if not layer.params or not layer.grads:
                logger.debug(
                    "optimizer skipping layer with no params or grads",
                    layer=layer
                )
                continue
            for _, params, key, grad in self._recursive_items(
                layer.params,
                layer.grads
            ):
                params[key] -= self.lr * grad
    def zero_grad(self):
        """Clear gradients of all managed layers."""
        logger.debug("optimizer zero_grad", layer_count=len(self.layers))

        for layer in self.layers:
            layer.grads = {}

class Adam(Optimizer):
    def __init__(
        self,
        layers=None,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        super().__init__(layers, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        """Perform a single optimization step updating layer parameters."""
        logger.debug("optimizer step", layer_count=len(self.layers), lr=self.lr)

        self.t += 1

        for layer in self.layers:
            if not layer.params or not layer.grads:
                continue

            for path, params, key, grad in self._recursive_items(
                layer.params,
                layer.grads
            ):
                if path not in self.m:
                    self.m[path] = Array.zeros(grad.shape)
                    self.v[path] = Array.zeros(grad.shape)

                self.m[path] = (
                    self.beta1 * self.m[path]
                    + (1 - self.beta1) * grad
                )

                self.v[path] = (
                    self.beta2 * self.v[path]
                    + (1 - self.beta2) * (grad ** 2)
                )

                m_hat = self.m[path] / (1 - self.beta1 ** self.t)
                v_hat = self.v[path] / (1 - self.beta2 ** self.t)

                params[key] -= (
                    self.lr *
                    m_hat /
                    (v_hat ** 0.5 + self.epsilon)
                )

    def zero_grad(self):
        """Clear gradients of all managed layers."""
        logger.debug("optimizer zero_grad", layer_count=len(self.layers))

        for layer in self.layers:
            layer.grads = {}