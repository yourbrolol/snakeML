from structs import Array
from debug import get_logger, operation_context, collect_diagnostics, summarize_diagnostics, describe, current_context
from debug.validator import ensure_type
from debug.errors import OperationError

logger = get_logger(__name__)

class Loss:
    def __init__(self, name):
        self.name = name
    def forward(self, y_pred, y_true): raise NotImplementedError
    def backward(self, y_pred, y_true): raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        """Mean Squared Error criterion."""
        super().__init__("MSELoss")

    def forward(self, y_pred, y_true):
        """Compute the Mean Squared Error loss between predicted and ground truth values."""
        with operation_context("MSE.forward"):
            logger.debug("enter forward; y_pred=%s", summarize_diagnostics(y_pred))
            logger.debug("enter forward; y_true=%s", summarize_diagnostics(y_true))
            # runtime type check (useful when inputs may be plain lists)
            try:
                if not isinstance(y_true, Array):
                    y_true = Array(y_true)
                ensure_type(y_pred, Array, name="y_pred")
                ensure_type(y_true, Array, name="y_true")
            except Exception as e:
                # attach current debug context to the raised OperationError
                ctx = current_context()
                logger.error("validation failed: %s", e)
                raise OperationError("MSE.forward validation failed", context=ctx, details={"error": str(e)})

            try:
                out = 0.5 * (y_pred - y_true) ** 2
            except Exception as e:
                logger.error("forward computation error: %s", e)
                raise

            logger.debug("forward result: %s", describe(out))
            logger.info("forward diagnostics: %s", summarize_diagnostics(out))
            return out

    def backward(self, y_pred, y_true):
        """Compute the gradient of MSE loss with respect to predictions."""
        with operation_context("MSE.backward"):
            logger.debug("enter backward; y_pred=%s", summarize_diagnostics(y_pred))
            try:
                if not isinstance(y_true, Array):
                    y_true = Array(y_true)
            except Exception as e:
                logger.error("backward input conversion failed: %s", e)
                raise

            grad = y_pred - y_true
            logger.debug("backward grad: %s", describe(grad))
            return grad

class CrossEntropy(Loss):
    """Cross-entropy loss for pre-softmaxed (probability) predictions.

    Expects y_pred to already be a probability distribution (i.e. softmax has
    been applied upstream).  y_true must be a one-hot encoded Array of the
    same shape.

    Forward
    -------
    L = -1/N * sum_i( y_true_i * log(y_pred_i) )

    Backward
    --------
    The gradient of softmax cross-entropy with respect to the *pre-softmax*
    logits simplifies to:

        dL/dz = (y_pred - y_true) / N

    This is the clean closed-form result you get when the softmax and
    cross-entropy are differentiated together.
    """

    def __init__(self):
        """Cross-entropy criterion (assumes softmax has already been applied)."""
        super().__init__("CrossEntropyLoss")

    def forward(self, y_pred, y_true):
        """Compute mean cross-entropy loss.

        Parameters
        ----------
        y_pred : Array
            Predicted probabilities (post-softmax), shape (N, C) or (C,).
        y_true : Array
            One-hot ground-truth labels, same shape as y_pred.

        Returns
        -------
        float
            Scalar mean cross-entropy loss.
        """
        with operation_context("CrossEntropy.forward"):
            logger.debug("enter forward; y_pred=%s", summarize_diagnostics(y_pred))
            logger.debug("enter forward; y_true=%s", summarize_diagnostics(y_true))

            try:
                if not isinstance(y_pred, Array):
                    y_pred = Array(y_pred)
                if not isinstance(y_true, Array):
                    y_true = Array(y_true)
                ensure_type(y_pred, Array, name="y_pred")
                ensure_type(y_true, Array, name="y_true")
            except Exception as e:
                ctx = current_context()
                logger.error("validation failed: %s", e)
                raise OperationError(
                    "CrossEntropy.forward validation failed",
                    context=ctx,
                    details={"error": str(e)},
                )

            try:
                # -sum(y_true * log(y_pred)) / N
                # clip probabilities away from 0 to guard against log(0)
                eps = 1e-12
                log_probs = (y_pred + eps).log()
                nll = y_true * log_probs          # element-wise
                loss = -nll.sum() / len(list(y_pred._flatten()))
            except Exception as e:
                logger.error("forward computation error: %s", e)
                raise

            logger.debug("forward loss: %s", loss)
            logger.info("forward diagnostics: %s", summarize_diagnostics(y_pred))
            return loss

    def backward(self, y_pred, y_true):
        """Compute the gradient of the fused softmax + cross-entropy loss.

        Parameters
        ----------
        y_pred : Array
            Predicted probabilities (post-softmax), shape (N, C) or (C,).
        y_true : Array
            One-hot ground-truth labels, same shape as y_pred.

        Returns
        -------
        Array
            Gradient w.r.t. the pre-softmax logits: (y_pred - y_true) / N.
        """
        with operation_context("CrossEntropy.backward"):
            logger.debug("enter backward; y_pred=%s", summarize_diagnostics(y_pred))

            try:
                if not isinstance(y_pred, Array):
                    y_pred = Array(y_pred)
                if not isinstance(y_true, Array):
                    y_true = Array(y_true)
            except Exception as e:
                logger.error("backward input conversion failed: %s", e)
                raise

            n = len(list(y_pred._flatten()))
            grad = (y_pred - y_true) / n
            logger.debug("backward grad: %s", describe(grad))
            return grad


class Softmax:
    """Numerically stable softmax, operating along the last axis.

    Numerics
    --------
    To prevent overflow in ``exp(x)`` the max-shift trick is applied
    per-sample before exponentiation::

        s(x)_i = exp(x_i - max(x)) / sum_j( exp(x_j - max(x)) )

    Input / output shapes
    ----------------------
    * **1-D** ``(C,)``  → ``(C,)``
    * **2-D** ``(N, C)`` → ``(N, C)`` — applied row-wise via ``Array.unbind``

    Backward
    --------
    Given upstream gradient ``g`` and softmax output ``p``, the exact
    per-sample Jacobian–vector product is::

        dx_i = p_i * (g_i - sum_j(g_j * p_j))

    which is equivalent to the full ``J^T g`` without explicitly building
    the C×C Jacobian matrix.
    """

    def __init__(self):
        self.name = "Softmax"
        self._output = None   # cache forward output for backward

    # ------------------------------------------------------------------
    # Internal: apply softmax to a single 1-D Array
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_1d(row):
        """Numerically stable softmax on a 1-D Array. Returns a 1-D Array."""
        m = row.max()                 # scalar
        shifted = row - m             # element-wise subtract max
        exp_x = shifted.exp()         # element-wise exp
        s = exp_x.sum()              # partition function (scalar)
        return exp_x / s

    @staticmethod
    def _softmax_grad_1d(p, g):
        """Jacobian-vector product for a single softmax output.

        Parameters
        ----------
        p : Array  — softmax probabilities, shape (C,)
        g : Array  — upstream gradient, shape (C,)

        Returns
        -------
        Array  shape (C,)
        """
        dot = (p * g).sum()           # scalar: sum_j(g_j * p_j)
        return p * (g - dot)          # p_i * (g_i - dot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x):
        """Apply softmax along the last axis.

        Parameters
        ----------
        x : Array
            Logits of shape ``(C,)`` or ``(N, C)``.

        Returns
        -------
        Array
            Probabilities of the same shape as *x*.
        """
        with operation_context("Softmax.forward"):
            if not isinstance(x, Array):
                x = Array(x)
            logger.debug("enter forward; x=%s", summarize_diagnostics(x))

            if x.ndim == 1:
                out = self._softmax_1d(x)
            elif x.ndim == 2:
                rows = x.unbind(axis=0)           # list of 1-D Arrays
                out = Array.stack(
                    [self._softmax_1d(r) for r in rows],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"Softmax expects 1-D or 2-D input, got shape {x.shape}"
                )

            self._output = out
            logger.debug("forward out: %s", summarize_diagnostics(out))
            return out

    def backward(self, grad):
        """Compute the Jacobian–vector product of softmax w.r.t. its input.

        Parameters
        ----------
        grad : Array
            Upstream gradient, same shape as the forward output.

        Returns
        -------
        Array
            Gradient w.r.t. the softmax input *x*, same shape.
        """
        with operation_context("Softmax.backward"):
            if self._output is None:
                raise RuntimeError("backward() called before forward()")
            if not isinstance(grad, Array):
                grad = Array(grad)

            p = self._output
            logger.debug("enter backward; grad=%s", summarize_diagnostics(grad))

            if p.ndim == 1:
                dx = self._softmax_grad_1d(p, grad)
            else:
                ps = p.unbind(axis=0)
                gs = grad.unbind(axis=0)
                dx = Array.stack(
                    [self._softmax_grad_1d(pi, gi) for pi, gi in zip(ps, gs)],
                    axis=0,
                )

            logger.debug("backward dx: %s", describe(dx))
            return dx