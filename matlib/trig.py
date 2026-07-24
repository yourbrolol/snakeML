from matlib import exp

def tanh(x):
    """Compute the hyperbolic tangent of x."""
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))