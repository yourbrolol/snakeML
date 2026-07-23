from structs import Array
from debug import get_logger, operation_context, collect_diagnostics, summarize_diagnostics, describe, current_context
from debug.validator import ensure_type
from debug.errors import OperationError

logger = get_logger(__name__)

class MSE:
    def __init__(self):
        """Mean Squared Error criterion."""
        self.name = "MSELoss"

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
