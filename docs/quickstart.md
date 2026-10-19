# Quick start

## Install

```bash
git clone https://github.com/yourbrolol/SnakeML.git
cd SnakeML
python -m pip install .
```

Dependencies: **none** (`pytest` for running tests).

## Train your first model

Everything lives in plain packages — add the repo root to your path (or `pip install -e .`) and go:

```python
from structs import Array
from basic import Sequential, Linear, ReLU
from basic.criterions import MSE
from basic.optimizers import Adam

# tiny regression problem: y = x0 + x1
X = [
    [0, 0], [0, 1], [1, 0], [1, 1], [2, 3], [4, 1],
]
Y = [[0], [1], [1], [2], [5], [5]]

model = Sequential([Linear(2, 4), ReLU(), Linear(4, 1)])
opt = Adam(model.layers, lr=0.01)
crit = MSE()

for step in range(500):
    opt.zero_grad()
    y_pred = model.forward(X)
    loss = crit.forward(y_pred, Array(Y))
    grad = crit.backward(y_pred, Array(Y))
    model.backward(grad)
    opt.step()

print(model.forward([3, 4]))   # ~ [[7]]
```

!!! note
    Pass `model.layers` — not `model` — to the optimizer. `Sequential` doesn't
    implement `params`/`grads`; the optimizer reads them off each `Layer`.

## What just happened

1. `model.forward(X)` walks the layers left-to-right.
2. `crit.backward` produces the gradient of the loss w.r.t. predictions (e.g. `MSE` → `(y_pred - y_true)`).
3. `model.backward(grad)` walks the layers right-to-left, filling each layer's `grads` dict.
4. `opt.step()` updates `params[key] -= lr * grad` (recursively, so nested dicts work too).
5. `opt.zero_grad()` clears `layer.grads = {}` at the top of each iteration.

## Using the Array

`Array` behaves like a mini NumPy: NumPy-style indexing, broadcasting arithmetic, reshaping, reductions, linalg:

```python
from structs import Array

a = Array([[1, 2], [3, 4]])
a.shape          # (2, 2)
a[1, 0]          # 3
a[:, 1]          # Array([[2], [4]])
a + a            # elementwise, NumPy-style broadcasting
a.matmul(a)      # batched matrix multiply
a.sum(axis=0)    # Array([4, 6])
a.reshape(-1)    # Array([[1], [2], [3], [4]])
Array.zeros((2, 3))
Array.randn((2, 2))
```

## Contribute

Contribute to SnakeML today on GitHub! See the repo's `tests/` for runnable examples and [Quick start (dev)](quickstart-dev.md) for the internals tour.
