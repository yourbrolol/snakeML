# SnakeML

**SnakeML** is the ML library made in Python and *for* Python. It runs through the plain Python interpreter, trading speed for simplicity, and provides generalized solutions that you can later shape for yourself.

Python 3.9+ · zero hard dependencies (`pytest`/`mypy` optional for dev).

## Architecture

SnakeML mostly consists of a few simple types that branch into functional classes:

| Type | Role |
|---|---|
| `Array` (`structs`) | Powers tensors and their operations |
| `matlib` | Pure-functional math kernels on nested lists (linalg, broadcasting, reductions, RNG) |
| `Layer` (`basic`) | Powers `forward` / `backward` passes, stores `params` and `input` |
| `Criterion` (`basic`) | Powers losses and gradients |
| `Activation(Layer)` | Powers nonlinear activations |
| `Sequential` (`basic`) | Powers neural networks |
| `Optimizer` (`basic`) | Powers gradient descent (`SGD`, `Adam`) |
| `debug` | Cross-cutting logging, validation, error hierarchy, operation contexts |

**This means that YOU can implement your own solutions if you haven't found one in the library. The base is here — the task is up to you.**

## Where to go next

- [Quick start](quickstart.md) — install and train your first model in ~20 lines.
- [Quick start (dev)](quickstart-dev.md) — a basic-to-advanced walkthrough through the project files and an explanation of how it all works.
- [API reference](api.md) — condensed reference for `structs`, `matlib`, and `basic`.

Read the full walkthrough verbatim on the [Quick start (dev)](quickstart-dev.md) page.
