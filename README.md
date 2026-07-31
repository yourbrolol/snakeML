# yourbrolol/SnakeML - the ML library made on Python and for Python.
## SnakeML was designed to be ran through the Python interpreter, trading speed for simplicity. SnakeML also provides generalized solutions that users can later shape for themselves.
## SnakeML dependencies (**1**):
- pytest.
**Some dependencies are not necessary. Required ones are in bold.**
## SnakeML architecture:
SnakeML mostly consists of a few simple types that branch into functional classes:
- `Layer` (powers `forward` / `backward` passes, stores `params` and `input`);
- `Criterion` (powers losses and gradients);
- `Activation(Layer)` (powers nonlinear activations);
- `Sequential` (powers neural networks);
- `Optimizer` (powers gradient descent);
- `Array` (powers tensors and their operations).
**This means that YOU can implement your own solutions if you haven't found one in the library. The base is here - the task is up to you.**
**Contribute to SnakeML today on GitHub!**
## WARNING: SnakeML is no longer under active development. It began as a learning project to better understand how modern machine learning frameworks work internally, and it has achieved that goal. Bug fixes and community contributions are still welcome, and pull requests will be reviewed when time permits.
