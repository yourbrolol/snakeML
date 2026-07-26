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
                loss = -nll.sum() / len(list(y_pred.flatten()))
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

            n = len(list(y_pred.flatten()))
            grad = (y_pred - y_true) / n
            logger.debug("backward grad: %s", describe(grad))
            return grad