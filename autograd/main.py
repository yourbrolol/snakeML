from structs import Array

from .mixins import ValueMixins


class Value(ValueMixins):
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        print(topo)

        self.grad = 1.0  # base-case
        for node in reversed(topo):
            if node.requires_grad:
                node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data})"


x, w, b = Value(1, False), Value(3, True), Value(5, True)
y = w * x + b
y.backward()
print(w.grad, x.grad, b.grad)
