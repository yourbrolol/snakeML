from debug import get_logger

logger = get_logger(__name__)

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
                logger.debug("optimizer skipping layer with no params or grads", layer=layer)
                continue
            for key in layer.params:
                if key in layer.grads: layer.params[key] -= self.lr * layer.grads[key]
    def zero_grad(self):
        """Clear gradients of all managed layers."""
        logger.debug("optimizer zero_grad", layer_count=len(self.layers))
        for layer in self.layers:
            if not layer.grads:
                continue
            layer.grads = {}
