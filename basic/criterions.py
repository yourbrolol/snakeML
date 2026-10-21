"""Loss (criterion) functions measuring prediction quality.

Each :class:`Loss` subclass provides ``forward`` (a scalar or array loss)
and ``backward`` (the gradient w.r.t. predictions). Operations are
wrapped in :func:`debug.operation_context` so failures surface with
ambient debug context attached.
"""

from __future__ import annotations

from debug import (
    collect_diagnostics,
    current_context,
    describe,
    get_logger,
    operation_context,
    summarize_diagnostics,
)
from debug.errors import OperationError
from debug.validator import ensure_type
from structs import Array

logger = get_logger(__name__)

__all__ = ["CrossEntropy", "Loss", "MSE"]


class Loss:
    """Base class for all loss criteria.

    Attributes:
        name: Human-readable criterion label.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def forward(self, y_pred: Array, y_true: Array) -> Array:
        raise NotImplementedError

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error criterion."""

    def __init__(self) -> None:
        super().__init__("MSELoss")

    def forward(self, y_pred: Array, y_true: Array) -> Array:
        """Compute the Mean Squared Error loss between predicted and ground truth values.

        Args:
            y_pred: Predicted values, :class:`~structs.Array`.
            y_true: Ground-truth values; lists are coerced to ``Array``.

        Returns:
            Element-wise ``0.5 * (y_pred - y_true) ** 2``.

        Raises:
            OperationError: When input validation fails; carries the
                ambient debug context and the underlying error.
        """
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
                raise OperationError(
                    "MSE.forward validation failed",
                    context=ctx,
                    details={"error": str(e)},
                )

            try:
                out = 0.5 * (y_pred - y_true) ** 2
            except Exception as e:
                logger.error("forward computation error: %s", e)
                raise

            logger.debug("forward result: %s", describe(out))
            logger.info("forward diagnostics: %s", summarize_diagnostics(out))
            return out

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        """Compute the gradient of MSE loss with respect to predictions.

        Args:
            y_pred: Predicted values, :class:`~structs.Array`.
            y_true: Ground-truth values; lists are coerced to ``Array``.

        Returns:
            ``y_pred - y_true``, the analytic MSE gradient.
        """
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

    def __init__(self) -> None:
        """Cross-entropy criterion (assumes softmax has already been applied)."""
        super().__init__("CrossEntropyLoss")

    def forward(self, y_pred: Array, y_true: Array) -> float:
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

        Raises:
            OperationError: When input validation fails; carries the
                ambient debug context and the underlying error.
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
                nll = y_true * log_probs  # element-wise
                loss = -nll.sum() / len(list(y_pred.flatten()))
            except Exception as e:
                logger.error("forward computation error: %s", e)
                raise

            logger.debug("forward loss: %s", loss)
            logger.info("forward diagnostics: %s", summarize_diagnostics(y_pred))
            return loss

    def backward(self, y_pred: Array, y_true: Array) -> Array:
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
