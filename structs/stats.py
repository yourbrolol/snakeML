import math

def sum(flat):
    """Returns the sum of all elements."""
    return sum(flat)

def mean(flat):
    """Returns the mean of all elements."""
    return sum(flat) / len(flat)

def std(flat):
    """Returns the standard deviation of all elements."""
    m = mean(flat)
    variance = sum((x - m) ** 2 for x in flat) / len(flat)
    return math.sqrt(variance)
