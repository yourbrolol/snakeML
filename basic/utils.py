from debug import get_logger, operation_context, summarize_diagnostics, describe
from debug.validator import ensure_type

logger = get_logger(__name__)

def im2col(input, kernel, stride):
    with operation_context("im2col", kernel=kernel, stride=stride):
        logger.debug("enter im2col; input=%s", summarize_diagnostics(input))
        try:
            ensure_type(input, object, name="input")  # replace with Array type if available
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
            logger.warning("kernel larger than input: input_shape=%s, kernel=%s", describe(input.shape), kernel)
            return []

        logger.debug("iterating y in 0..%d step %d, x in 0..%d step %d", y_max-1, SH, x_max-1, SW)
        for y in range(0, y_max, SH):
            for x in range(0, x_max, SW):
                patch = input[:, y:y+KH, x:x+KW]
                logger.debug("patch at y=%d x=%d -> %s", y, x, summarize_diagnostics(patch))
                patches.append(patch)
        logger.info("extracted %d patches", len(patches))
        return patches
