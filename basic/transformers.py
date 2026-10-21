"""Transformer-style building blocks: embeddings, normalization and attention.

Supplements :mod:`basic.layers` with pieces used by transformer
architectures: token/position :class:`Embedding`, :class:`LayerNorm`,
:class:`Residual` wrapping and scaled dot-product :class:`Attention`
with hand-derived backward passes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

from basic.activations import Softmax
from basic.optimizers import SGD
from structs import Array

from .layers import Layer

__all__ = ["Attention", "Embedding", "LayerNorm", "PosEmbedding", "Residual"]


class Embedding(Layer):
    """Token embedding: learned lookup table ``w`` indexed by integer tokens.

    Attributes:
        params["w"]: Embedding table of shape ``(vocab_size, d_model)``.
        grads["w"]: Scattered token gradients accumulated by ``backward``.
        _last_token: Fallback token index reused when the forward input
            is unavailable at backward time.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        table: Optional[Any] = None,
    ) -> None:
        """Build the table from ``table`` when given, else from ``Array.randn``."""
        super().__init__()
        if table is not None:
            self.params["w"] = Array.wraparray(table)
        else:
            self.params["w"] = Array.randn((vocab_size, d_model))
        self.grads["w"] = Array.zeros((vocab_size, d_model))

    def forward(self, x: Any) -> Any:
        """Look up rows of ``w`` for each token in ``x``.

        Args:
            x: Integer token indices (scalar or collection).

        Returns:
            The embedding vectors corresponding to ``x``.
        """
        self.input = x
        return self.params["w"][x]

    def backward(self, grad: Any) -> None:
        """Scatter upstream gradients into ``grads["w"]`` at the input tokens.

        Args:
            grad: Upstream gradients, one per token in :attr:`input`.
                    When :attr:`input` is unwritable, falls back to
                    :attr:`_last_token` repeated, then to ``range(len(grad))``.

        Returns:
            ``None``; gradients accumulate on the layer.
        """
        if self.input is not None:
            tokens = list(self.input)
        elif hasattr(self, "_last_token"):
            tokens = [self._last_token] * len(grad)
        else:
            tokens = list(range(len(grad)))
        self._last_token = tokens[-1] if tokens else 0
        for token, g in zip(tokens, grad):
            self.grads["w"][token] += g
        return None


class PosEmbedding(Layer):
    """Positional embedding: adds learned row ``w[pos]`` to the input.

    Attributes:
        params["w"]: Position table of shape ``(ctx_len, d_model)``.
        grads["w"]: Scattered position gradients accumulated by ``backward``.
    """

    def __init__(
        self,
        ctx_len: int,
        d_model: int,
        table: Optional[Any] = None,
    ) -> None:
        """Build the table from ``table`` when given, else from ``Array.randn``."""
        super().__init__()
        if table is not None:
            self.params["w"] = Array.wraparray(table)
        else:
            self.params["w"] = Array.randn((ctx_len, d_model))
        self.grads["w"] = Array.zeros((ctx_len, d_model))

    def forward(self, x: Any, pos_x: Any) -> Any:
        """Add learned positional embeddings to ``x``.

        Args:
            x: The activations to offset.
            pos_x: Integer position indices selecting table rows.

        Returns:
            ``x + w[pos_x]``; ``pos_x`` is cached for ``backward``.
        """
        self.input = pos_x
        return x + self.params["w"][pos_x]

    def backward(self, grad: Any) -> Any:
        """Scatter upstream gradients into ``grads["w"]`` at the cached positions.

        Args:
            grad: Upstream gradient with one entry per cached position.

        Returns:
            ``grad`` unchanged (positions are inputs, not parameters).
        """
        for pos, g in zip(self.input, grad):
            self.grads["w"][pos] += g
        return grad


class LayerNorm(Layer):
    """Layer normalization over the last axis with learned scale/shift.

    Caches the normalized activations ``x_hat`` and inverse std so the
    backward pass can reuse the forward statistics.
    """

    def __init__(self, input_size: int) -> None:
        """Initialize ``w`` (scale) to ones and ``b`` (shift) to zeros."""
        super().__init__()
        self.params["w"] = Array.ones((input_size,))
        self.params["b"] = Array.zeros((input_size,))

    def forward(self, input: Any, eps: float = 1e-5) -> Any:
        """Normalize the last axis and apply the learned affine transform.

        Args:
            input: Input whose last axis is normalized.
            eps: Variance stabilizer inside the square root.

        Returns:
            ``w * x_hat + b`` where ``x_hat = (x - mean) / sqrt(var + eps)``.
        """
        self.input = Array.wraparray(input)
        x = self.input
        self.eps = eps
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = ((x - self.mean) ** 2).mean(axis=-1, keepdims=True)
        self.inv_std = 1 / (self.var + eps).sqrt()
        self.x_hat = (x - self.mean) * self.inv_std
        return self.params["w"] * self.x_hat + self.params["b"]

    def backward(self, grad: Any) -> Any:
        """Compute parameter and input gradients for layer normalization.

        Args:
            grad: Upstream gradient, same shape as the forward output.

        Returns:
            Gradient w.r.t. the (pre-normalization) input, following the
            standard LN backward: ``inv_std * (dx_hat - mean(dx_hat)
            - x_hat * mean(dx_hat * x_hat))``.
        """
        # parameter gradients
        self.grads["w"] = (grad * self.x_hat).sum(axis=tuple(range(grad.ndim - 1)))
        self.grads["b"] = grad.sum(axis=tuple(range(grad.ndim - 1)))
        # input gradient
        dx_hat = grad * self.params["w"]
        mean1 = dx_hat.mean(axis=-1, keepdims=True)
        mean2 = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        dX = self.inv_std * (dx_hat - mean1 - self.x_hat * mean2)
        return dX


class Residual(Layer):
    """Residual wrapper adding ``fn(x)`` back onto the (raw) input.

    Attributes:
        fn: Wrapped layer whose output forms the residual branch.
    """

    def __init__(self, fn: Layer) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Any, fx: Optional[Any] = None) -> Any:
        """Apply ``fn`` to ``x`` and return ``x + fx``.

        Args:
            x: The residual stream.
            fx: Precomputed residual branch; computed via ``fn.forward(x)``
                when omitted.

        Returns:
            ``x + fx``, the summed stream.
        """
        self.input = x
        if fx is None:
            fx = self.fn.forward(x)
        return x + fx

    def backward(self, grad: Any) -> Any:
        """Backprop through both residual branches.

        Args:
            grad: Upstream gradient.

        Returns:
            ``grad + fn.backward(grad)`` — the identity branch plus the
            branch gradient.
        """
        return grad + self.fn.backward(grad)


class Attention(Layer):
    """Scaled dot-product (single-head) attention with hand-derived backward.

    Attributes:
        params["w"]: ``Q``/``K``/``V`` projection matrices ``(d_model, d_model)``.
        grads["w"]: Matching gradient dicts populated by ``backward``.
        d_model: Embedding dimensionality, used as the scale factor.
        softmax: Cached :class:`~basic.activations.Softmax` used on scores.
    """

    def __init__(self, window_size: int, d_model: int) -> None:
        """Initialize the (unused) window size and random Q/K/V projections."""
        super().__init__()
        self.params["w"] = {
            "Q": Array.randn((d_model, d_model)),
            "K": Array.randn((d_model, d_model)),
            "V": Array.randn((d_model, d_model)),
        }
        self.grads["w"] = {
            "Q": Array.randn((d_model, d_model)),
            "K": Array.randn((d_model, d_model)),
            "V": Array.randn((d_model, d_model)),
        }
        self.d_model = d_model
        self.softmax = Softmax()

    def forward(self, X: Any) -> Any:
        """Compute the attention output ``softmax(QK^T / sqrt(d)) @ V``.

        Args:
            X: Input activations of shape ``(N, d_model)``.

        Returns:
            Attended values of shape ``(N, d_model)``; intermediate
            ``Q, K, V, scores, attention`` are cached for ``backward``.
        """
        Wq, Wk, Wv = self.params["w"]["Q"], self.params["w"]["K"], self.params["w"]["V"]
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        scores = (Q @ K.T) / math.sqrt(self.d_model)
        attention = self.softmax.forward(scores)
        self.input, self.Q, self.K, self.V, self.scores, self.attention = (
            X,
            Q,
            K,
            V,
            scores,
            attention,
        )
        return attention @ V

    def backward(self, grad: Any) -> Any:
        """Backpropagate through the attention computation.

        Populates ``grads["w"]`` for the ``Q``/``K``/``V`` projections and
        returns the gradient w.r.t. the input ``X``.

        Args:
            grad: Upstream gradient of the attended values.

        Returns:
            Gradient w.r.t. ``X``: the sum of the Q/K/V branch gradients.
        """
        dA = grad @ self.V.T
        dV = self.attention.T @ grad
        dS = self.softmax.backward(dA)
        dScores = dS / math.sqrt(self.d_model)
        dQ = dScores @ self.K
        dK = dScores.T @ self.Q
        dXq = dQ @ self.params["w"]["Q"].T
        dWq = self.input.T @ dQ
        dXk = dK @ self.params["w"]["K"].T
        dWk = self.input.T @ dK
        dXv = dV @ self.params["w"]["V"].T
        dWv = self.input.T @ dV
        dX = dXq + dXk + dXv
        self.grads["w"]["Q"], self.grads["w"]["K"], self.grads["w"]["V"] = dWq, dWk, dWv
        return dX


if __name__ == "__main__":
    att = Attention(3, 4)
    print(att.forward(Array.randn((3, 4))))
    print(att.backward(Array.randn((3, 4))))
