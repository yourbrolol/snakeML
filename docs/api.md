# API reference

Condensed, deliberately non-exhaustive — the docstrings in the source are richer.

## structs — `Array`

```python
from structs import Array
```

| Area | Members |
|---|---|
| Creation | `Array(data)` · `zeros(shape)` `ones(shape)` `empty(shape)` `full(shape, value)` `eye(size, k)` `arange(start, stop, step)` `linspace(start, stop, num, shape)` `randn(shape)` `randint(low, high, shape)` `.clone()` |
| Arithmetic | `+ - * / // % **` (+ reflected & in-place dunders) · comparisons → boolean arrays · `@` / `matmul` |
| Linalg | `dot(axes)` (tensordot) `matmul` `matvec` `inner` `outer` `inverse` `det` `solve` `cholesky` `trace` `diag` `cross` |
| Unary | `exp log sqrt square abs sin cos tan tanh floor ceil round clip sign` ... |
| Shape | `reshape(*shape)` `squeeze` `unsqueeze` `expand_dims` `swapaxes` `moveaxis` `permute` `transpose` `T` `broadcast_to` `roll` `pad` |
| Manipulate | `concatenate` `stack` `split` `unbind` `vstack` `hstack` `flip` `take` `put` `gather` `scatter` `masked_fill` `where` `fill` `astype` |
| Stats | `sum mean var std max min prod argmax argmin count_nonzero norm` — all `(axis=None, keepdims=False)` |
| Sampling | `shuffle` `permutation` `choice(size)` `searchsorted` `unique` `sort` `argsort` |
| Meta | `.data .shape .ndim .size .dtype (label) .tolist() .flatten()` |

## matlib

```python
import matlib
```

- **broadcasting**: `broadcast(a, b)`, `elementwise(a, b, op)`
- **linalg**: `matmul matvec outer tensordot transpose permute inverse solve det cross cholesky`
- **ops**: `exp log log2 log10 sqrt square abs sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil round clip sign`
- **stats**: `sum mean variance std max min prod argmax argmin count_nonzero norm` and `_axis` variants with `keepdims`
- **manipulation**: `concatenate stack split unbind flip`
- **rng**: `normal(mean=0.0, std=1.0)`

## basic

```python
import basic
```

- **Layers**: `Layer` · `Linear(input_dim, output_dim, initializer)` · `Conv2D(input_dim, output_dim, kernel, stride)` · `MaxPool2D` (placeholder)
- **Activations**: `LinearActivation ReLU GELU Sigmoid Tanh Softmax`
- **Losses**: `MSE()` · `CrossEntropy()` (expects softmax probs + one-hot)
- **Optimizers**: `SGD(layers, lr=0.01)` · `Adam(layers, lr=0.001, beta1, beta2, epsilon)` — pass `model.layers`
- **Networks**: `Sequential(layers)` — `forward`/`backward` only; optimization lives in the optimizer
- **Initializers** (functors): `Zeros Ones Normal Uniform Constant XavierNormal XavierUniform KaimingNormal KaimingUniform`
- **Transformers** (experimental): `Embedding` `PosEmbedding` `LayerNorm` `Residual` `Attention`
- `from basic.utils import im2col`

## debug

- `debug.config` — `configure configure_from_env set_level is_enabled get_config reset_config`
- `debug.logger` — `get_logger(name)`, `Logger.debug/info/warning/error/critical`, `Logger.bind(**ctx)`
- `debug.context` — `operation_context(name)` context manager, `current_context`, `update_context`, `clear_context`
- `debug.errors` — `LibraryError → ValidationError → ShapeError / TypeMismatchError`, `OperationError`
- `debug.validator` — `ensure_type ensure_shape validate`
- `debug.diagnostics` — `collect_diagnostics summarize_diagnostics`
- `debug.formatter` — `format_message describe`
