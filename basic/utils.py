"""Shared helpers for layer implementations.

Currently hosts :func:`im2col`, the image-to-column patch extractor
backing :class:`basic.layers.Conv2D`'s im2col-based convolution.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from debug import describe, get_logger, operation_context, summarize_diagnostics
from debug.validator import ensure_type

logger = get_logger(__name__)

__all__ = ["im2col"]


def im2col(
    input: Any,
    kernel: Sequence[int],
    stride: Sequence[int],
) -> List[Any]:
    """Extract image patches according to kernel size and stride.

    Args:
        input: Image of shape ``(C, H, W)`` (patch extraction is
            channel-agnostic, spanning ``H`` and ``W`` only).
        kernel: Two-element ``[KH, KW]`` patch size.
        stride: Two-element ``[SH, SW]`` sampling stride for patch origins.

    Returns:
        A list of shape ``(C, KH, KW)`` patches ordered row-major over
        patch origins, or an empty list when the kernel is larger than
        the spatial extent of the input.
    """
    with operation_context("im2col", kernel=kernel, stride=stride):
        logger.debug("enter im2col; input=%s", summarize_diagnostics(input))
        try:
            ensure_type(
                input, object, name="input"
            )  # replace with Array type if available
            KH, KW = kernel
            SH, SW = stride
            H, W = input.shape[1], input.shape[2]  # assuming (C, H, W)
        except Exception as e:
            logger.error("input/setup validation failed: %s", e)
            raise

        patches = []
        y_max = H - KH + 1
        x_max = W - KW + 1
        if y_max <= 0 or x_max <= 0:
            logger.warning(
                "kernel larger than input: input_shape=%s, kernel=%s",
                describe(input.shape),
                kernel,
            )
            return []

        logger.debug(
            "iterating y in 0..%d step %d, x in 0..%d step %d",
            y_max - 1,
            SH,
            x_max - 1,
            SW,
        )
        for y in range(0, y_max, SH):
            for x in range(0, x_max, SW):
                patch = input[:, y : y + KH, x : x + KW]
                logger.debug(
                    "patch at y=%d x=%d -> %s", y, x, summarize_diagnostics(patch)
                )
                patches.append(patch)
        logger.info("extracted %d patches", len(patches))
        return patches
