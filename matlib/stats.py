import math

from debug import get_logger
from debug.errors import ValidationError

logger = get_logger(__name__)


def _sum_values(flat):
    """Accumulate and return the sum of flat elements."""
    total = 0
    for value in flat:
        total += value
    return total


def sum(flat):
    """Returns the sum of all elements."""
    return _sum_values(flat)


def mean(flat):
    """Returns the mean of all elements."""
    flat = list(flat)
    if not flat:
        logger.error("mean called on empty collection")
        raise ValidationError("mean of empty array")
    return sum(flat) / len(flat)


def std(flat):
    """Returns the standard deviation of all elements."""
    flat = list(flat)
    if not flat:
        logger.error("std called on empty collection")
        raise ValidationError("std of empty array")
    m = mean(flat)
    variance = sum((x - m) ** 2 for x in flat) / len(flat)
    return math.sqrt(variance)
