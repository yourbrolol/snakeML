from matlib import exp, tanh
from basic.layers import Layer
from debug import get_logger
from structs import Array
from math import pi, sqrt

logger = get_logger(__name__)

class Activation(Layer):
    def __init__(self, name):
        """Base activation layer initializing name and input state."""
        super().__init__()
        self.name = name
        self.input = None
        self.output = None
    def forward(self, x): raise NotImplementedError
    def backward(self, x): raise NotImplementedError

class LinearActivation(Activation):
    def __init__(self):
        """Identity activation layer."""
        super().__init__("LinearActivation")
    def forward(self, x):
        """Pass input through without transformation."""
        logger.debug("linear activation forward", activation=self.name, value=x)
        return x
    def backward(self, x):
        """Pass gradient through without transformation."""
        logger.debug("linear activation backward", activation=self.name, value=x)
        return x

class ReLU(Activation):
    def __init__(self):
        """Rectified Linear Unit activation layer."""
        super().__init__("ReLU")
    def forward(self, x):
        """Apply ReLU activation element-wise."""
        self.input = x
        logger.debug("relu forward", value=x)
        out = max(0, x)
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        """Compute backward pass for ReLU activation."""
        logger.debug("relu backward", grad=grad, input=self.input)
        return grad * (self.input > 0)

class GELU(Activation):
    def __init__(self):
        """Gaussian Error Linear Unit activation layer."""
        super().__init__("GELU")
    def forward(self, x):
        """Apply GELU activation element-wise."""
        self.input = x
        logger.debug("gelu forward", value=x)
        out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        """Compute backward pass for GELU activation."""
        logger.debug("gelu backward", grad=grad, input=self.input)

        x = self.input
        k = (2 / pi) ** 0.5

        u = k * (x + 0.044715 * x ** 3)
        tanh_term = tanh(u)
        sech2_term = 1 - tanh_term ** 2

        gelu_grad = (
            0.5 * (1 + tanh_term)
            + 0.5 * x * sech2_term * k * (1 + 0.134145 * x ** 2)
        )

        return grad * gelu_grad

class Sigmoid(Activation):
    def __init__(self):
        """Sigmoid activation layer."""
        super().__init__("Sigmoid")
    def forward(self, x):
        """Apply Sigmoid activation element-wise."""
        self.input = x
        logger.debug("sigmoid forward", value=x)
        out = 1 / (1 + exp(-x))
        self.output = out
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        """Compute backward pass for Sigmoid activation."""
        logger.debug("sigmoid backward", grad=grad, input=self.input)
        out = self.output
        return grad * (out*(1-out))

class Tanh(Activation):
    def __init__(self):
        """Tanh activation layer."""
        super().__init__("Tanh")
    def forward(self, x):
        """Apply Tanh activation element-wise."""
        self.input = x
        logger.debug("tanh forward", value=x)
        e_px, e_nx = exp(x), exp(-x)
        out = (e_px-e_nx) / (e_px+e_nx)
        self.output = out
        return out if isinstance(out, Array) else Array([out])
    def backward(self, grad):
        """Compute backward pass for Tanh activation."""
        logger.debug("tanh backward", grad=grad, input=self.input)
        return grad * (1-(self.output**2))

class Softmax(Activation):
    """Numerically stable softmax activation.

    Applies softmax along *axis* (default: last axis).
    Input / output shapes:
      * 1-D (C,)    → (C,)
      * 2-D (N, C)  → (N, C)  applied row-wise
    """

    def __init__(self):
        """Softmax activation layer."""
        super().__init__("Softmax")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_1d(row):
        """Numerically stable softmax on a 1-D Array."""
        m = row.max()           # subtract max to prevent exp overflow
        e = (row - m).exp()
        return e / e.sum()

    @staticmethod
    def _grad_1d(p, g):
        """Jacobian–vector product for a single softmax output.

        dx_i = p_i * (g_i - sum_j(g_j * p_j))
        """
        return p * (g - (p * g).sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x, axis=-1):
        """Apply softmax along *axis*.

        Parameters
        ----------
        x : Array
            Logits of shape (C,) or (N, C).
        axis : int
            Axis along which softmax is computed (default -1, i.e. last).

        Returns
        -------
        Array  same shape as x.
        """
        if not isinstance(x, Array):
            x = Array(x)
        self.input = x
        logger.debug("softmax forward", activation=self.name, shape=x.shape)

        if x.ndim == 1:
            out = self._softmax_1d(x)
        elif x.ndim == 2:
            # apply row-wise along axis=1 (last axis)
            out = Array.stack(
                [self._softmax_1d(row) for row in x.unbind(axis=0)],
                axis=0,
            )
        else:
            raise ValueError(
                f"Softmax supports 1-D and 2-D inputs, got shape {x.shape}"
            )

        self.output = out
        logger.debug("softmax forward done", activation=self.name, shape=out.shape)
        return out

    def backward(self, grad):
        """Compute gradient of softmax w.r.t. its input.

        Uses the compact Jacobian–vector product form instead of the full
        C×C Jacobian: dx = p * (g - dot(g, p))

        Parameters
        ----------
        grad : Array
            Upstream gradient, same shape as forward output.

        Returns
        -------
        Array  same shape as grad.
        """
        if not isinstance(grad, Array):
            grad = Array(grad)
        p = self.output
        logger.debug("softmax backward", activation=self.name)

        if p.ndim == 1:
            return self._grad_1d(p, grad)

        return Array.stack(
            [self._grad_1d(pi, gi)
             for pi, gi in zip(p.unbind(axis=0), grad.unbind(axis=0))],
            axis=0,
        )