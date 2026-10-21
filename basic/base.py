"""Lowest-level neuron building blocks (legacy/experimental).

:class:`Neuron` and its :class:`LinearNeuron` subclass operate directly
on :class:`~structs.Vector` objects rather than :class:`~structs.Array`;
:class:`NeuronBasedLinear` assembles them into a vector-output unit.
"""

from __future__ import annotations

from typing import Any, List, Optional

from structs import Vector

__all__ = ["LinearNeuron", "Neuron", "NeuronBasedLinear"]


class Neuron:
    """A single neuron with weight/bias vectors and gradient slots.

    Attributes:
        w: Weight vector.
        bias: Bias vector.
        input: Cached forward input.
        d_w: Cached gradient w.r.t. ``w``.
        d_b: Cached gradient w.r.t. ``bias``.
    """

    def __init__(self, input_dim: int) -> None:
        self.w: Vector = Vector()
        self.bias: Vector = Vector()
        self.input: Optional[Any] = None
        self.d_w: Optional[Any] = None
        self.d_b: Optional[Any] = None

    def forward(self, input_data: Any) -> Any:
        raise NotImplementedError

    def backward(self, output_grad: Any) -> Any:
        raise NotImplementedError


class LinearNeuron(Neuron):
    """Affine neuron: ``input * w + b``.

    The two-argument constructor mirrors :class:`Neuron` but allocates
    constant-initialized weight/bias vectors of the requested size.
    """

    def __init__(self, input_dim: int) -> None:
        self.w: Vector = Vector([0.1] * input_dim)
        self.bias: Vector = Vector(0.1)
        self.input: Optional[Any] = None
        self.d_w: Optional[Any] = None
        self.d_b: Optional[Any] = None

    def forward(self, input_data: Any) -> Any:
        """Compute the affine transformation ``input_data * w + b``.

        Args:
            input_data: Vector of inputs matching ``len(self.w)``.

        Returns:
            The neuron's scalar activation.
        """
        self.input = input_data
        return input_data * self.w + self.b

    def backward(self, output_grad: Any) -> Any:
        """Accumulate weight/bias gradients and back-propagate to the input.

        Args:
            output_grad: Upstream gradient of this neuron's output.

        Returns:
            Gradient w.r.t. the input, ``output_grad * w``.
        """
        self.d_w = output_grad / self.input
        self.d_b = output_grad
        return output_grad * self.w


class NeuronBasedLinear:
    """A linear unit built from ``output_dim`` individual neurons.

    Attributes:
        neurons: One :class:`Neuron` per output unit.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.neurons: List[Neuron] = [Neuron(input_dim) for _ in range(output_dim)]

    def forward(self, input_data: Any) -> Vector:
        """Run every neuron on the input and stack their activations.

        Args:
            input_data: Shared input vector for all neurons.

        Returns:
            A :class:`~structs.Vector` of per-neuron activations.
        """
        return Vector([neuron.forward(input_data) for neuron in self.neurons])

    def backward(self, output_gradient: Any) -> Vector:
        """Sum each neuron's upstream contribution into the input gradient.

        Args:
            output_gradient: Per-neuron gradients addressable by index.

        Returns:
            The accumulated gradient w.r.t. the layer input.
        """
        input_gradient = Vector([0] * len(self.neurons[0].w))

        for i, neuron in enumerate(self.neurons):
            input_gradient += neuron.backward(output_gradient[i])

        return input_gradient
