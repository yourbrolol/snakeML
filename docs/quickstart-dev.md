# Quick start (dev)

This is a walkthrough through the SnakeML codebase, from the bottom layer up. Read it alongside the source — each section points at concrete files.

## The map

```
mllib/
├── matlib/        pure math on nested lists (no class dispatch)   ← layer 1
├── structs/       the Array container, composed from mixins      ← layer 2
├── basic/         NN layers, losses, optimizers                  ← layer 3
├── debug/         logging / validation / errors (cross-cutting)
├── localdev/      scratch demo & experiment scripts
└── tests/         pytest + unittest
```

Dependencies flow one way: `basic` → `structs` → `matlib`, with `debug` used by all of them.

---

## Level 0 — `debug/`: the nervous system

Small, mostly-independent modules you should read first because *everything* imports them:

| File | What it gives |
|---|---|
| `debug/config.py` | Module-level config dict: `configure(**overrides)`, `set_level(...)`, `is_enabled(...)`, env-var config |
| `debug/logger.py` | Custom `Logger` with `debug/info/warning/error/critical`; `get_logger(name)`; `bind(**ctx)` for preset context |
| `debug/context.py` | `operation_context(name, **meta)` — a context manager that installs ambient metadata ("what op am I in") |
| `debug/errors.py` | `LibraryError(root) → ValidationError → ShapeError / TypeMismatchError`, `OperationError(context, details)` |
| `debug/validator.py` | `ensure_type(...)`, `ensure_shape(...)` — raise the exceptions above |
| `debug/formatter.py` | `[snakeML]`-prefix formatting, `describe(value)` — one-line human-readable object dump |
| `debug/diagnostics.py` | JSON-ish runtime snapshots for log lines |

The pattern used everywhere in `basic/`:

```python
from debug.context import operation_context
from debug.errors import OperationError

with operation_context("linear.forward", layer="Linear"):
    try:
        ...           # risky math
    except Exception as e:
        raise OperationError(context=current_context(), details=str(e))
```

So when something fails deep in the network, the error carries the *ambience* of where it happened.

---

## Level 1 — `matlib/`: dtype-agnostic math kernels

`matlib` functions take and return **plain nested lists** (or `Array`s — they unwrap duck-typed data). There is no dispatch class here; it's the functional core.

- `matlib/broadcasting.py` — the heart: `broadcast(a, b)` implements NumPy broadcast semantics (with an extra lenient 1-D-vs-2-D alignment), and `elementwise(a, b, op)` combines broadcasting with arbitrary binary callables. **Every `a + b`, `a * b`, comparison etc. on `Array` funnels through here.**
- `matlib/linalg.py` — `transpose`, `matmul` (batched over leading axes), `matvec`, `outer`, `tensordot` (general axis-contraction — used by `Conv2D`), `inverse` (Gauss-Jordan), `solve`, `det`, `cross`, `cholesky`, `permute`.
- `matlib/ops.py` — unary elementwise kernels: `exp, log, sqrt, tanh, clip, ...` via the recursive `_apply_unary`.
- `matlib/stats.py` — two tiers: global reductions over flat iterables (`sum`, `mean`, `variance`, ...) and axis-aware versions (`sum_axis(arr, axis, keepdims)`).
- `matlib/manipulation.py` — `concatenate`, `stack`, `split`, `unbind`, `flip`.
- `matlib/rng.py` — the entire randomness source: `normal(mean, std)` via Box–Muller on `random.random()`.
- `matlib/operators.py`, `matlib/reductions.py` — reserved placeholder namespaces.
- `matlib/__init__.py` — re-exports ~42 names (intentionally shadowing builtins `abs/sum/min/max`).

Run the tests as you read; they're the executable spec of all this.

---

## Level 2 — `structs/`: the Array

### `structs/dtypes.py`

Not a dtype system — it's the shared **type-alias vocabulary** used by every signature: `Scalar`, `NestedList`, `ArrayLike`, `IndexKey`, `Shape`, `Strides`, plus a TypeVar-based 3.9-compatible `Self` fallback (`ArrayT`/`ReturnSelf`).

### `structs/array/protocol.py` — the contract

```python
class ArrayProtocol(Protocol):
    data: NestedList; shape: Shape; strides: Strides; offset: int; ndim: int
    ...
```

This is the key design pattern: each behavioral mixin subclasses `ArrayProtocol`, so mypy type-checks each mixin independently against the shared contract, and `Array` satisfies it simply by inheriting the mixin composition.

### `structs/array/mixins.py` — the composition

```python
class ArrayMixins(ArrayConstructors, ArrayUtilsOps, ArrayShapeOps, ArrayMathOps, ArrayStatsOps):
    pass
```

`ArrayArray` (sic: `Array` in `structs/array/array.py`) is `class Array(ArrayMixins)` — a *pure shell*. Attributes are computed eagerly at construction (`_to_list` + `_get_shape` + `_strides`) and recomputed after every in-place mutation; there are no views yet (`offset` is reserved for that).

### The four behavioral mixins

| Mixin | File | Highlights |
|---|---|---|
| Constructors | `constructors.py` | `zeros/ones/eye/full/arange/linspace/randn/tile/...` — classmethod factories filling per-element over a shape grid |
| Utils | `utils.py` (~875 lines, the big one) | internal shape machinery (`_to_list`, `_get_shape`, `_strides`, `_wrap_result` via `type(self)`), plus `flatten/tolist/item`, `concatenate/stack/split/unbind`, `where/take/put/gather/scatter/masked_fill/astype`, `unique/sort/argsort/searchsorted/shuffle/choice` |
| Math | `ops.py` (~910 lines) | full dunder surface (`+ - * / // % ** & \| ^ abs` + `r*`'s + `i*`'s), all routed through `_elementwise(...)` → `matlib.broadcasting.elementwise`; linalg delegation (`dot` via `tensordot`, `matmul`, `outer`, `inverse`, ...); unary delegation to `matlib.ops` |
| Shape | `shape.py` | `reshape` (single `-1` inference), `squeeze/unsqueeze`, `swapaxes/moveaxis/permute/transpose`, `T`, `broadcast_to`, `roll`, `pad` (2-D) |
| Stats | `stats.py` | `sum/mean/var/std/max/min/prod/argmax/argmin/norm` all with `(axis=None, keepdims=False)` |

In-place ops go through `_update_self(result)`, which rebinds `data/shape/ndim`.

### `structs/array/indexing.py` and `stats.py`

- `indexing.py` — the NumPy-style indexing engine: `_normalize_key` validates `int / slice / bool list / Array / ...`, expands Ellipsis, pads missing trailing keys as full slices; `_getitem` / `_setitem` run over it. Slicing is garaunteed by tests like `arr[:, 1]`.
- `stats.py` — thin `(axis, keepdims)` adapters calling `matlib.stats.*_axis`.

Key things to know before hacking on the Array:

1. Everything rebuilds via `type(self)` — subclassing keeps fluent returns.
2. `dtype` is an inferred *string label* on demand (`astype` casts to native Python types); there is no native-typed backing store.
3. MRO gotcha: `ArrayConstructors.repeat`/`tile` (classmethods) shadow same-named instance methods in `ArrayShapeOps` — those instances are currently dead code.

---

## Level 3 — `basic/`: layers, losses, optimizers

### `basic/layers.py` — the core abstraction

```python
class Layer:
    params: Dict[str, Any]      # e.g. {"w": Array, "b": Array}
    grads: Dict[str, Any]       # filled by backward, same keys
    submodules: Dict[str, Layer]
    input: Any                  # cached for backward

    def forward(self, input_data): ...
    def backward(self, output_grad): ...     # returns d(input)
    def parameters(self): ...                # recursive, dotted keys
```

`Linear(input_dim, output_dim, initializer=None)`:
- forward: `w.matmul(input) + b` under `operation_context("linear.forward")`
- backward: `grads["w"] = grad.outer(self.input)`, `grads["b"] = grad`, returns `w.T.matmul(grad)`

`Conv2D(input_dim, output_dim, kernel, stride)` uses `basic/utils.im2col` (patch extraction via `input[:, y:y+KH, x:x+KW]` slice tricks) + `tensordot` over the last three axes. (`MaxPool2D` is a placeholder.)

### `basic/activations.py`

All subclasses of `Activation(Layer)`: `ReLU`, `Sigmoid`, `Tanh`, `GELU` (tanh approx + hand-derived backward), `Softmax` (max-subtracted for stability; supports 1-D and row-wise 2-D; compact Jacobian backward `p*(g - (p*g).sum())`), `LinearActivation` (identity). Convention: forward coerces non-`Array` inputs and caches `self.input`/`self.output` for the derivative.

### `basic/criterions.py`

- `MSE`: forward `0.5*(y_pred-y_true)**2` (elementwise), backward `(y_pred - y_true)` — the 0.5 cancels the d²/dx².
- `CrossEntropy`: assumes pred is **already softmax probabilities** and targets are one-hot; backward `(y_pred - y_true)/N`. Heavily instrumented: `operation_context`, `ensure_type`, failures become `OperationError`.

### `basic/optimizers.py`

```python
class Optimizer:
    def __init__(self, layers=None, lr=0.01): ...
```

Pass `model.layers`. `step()` walks `(dotted_path, params_dict, key, grad)` via `_recursive_items` — so a nested `params["w"]["Q"]` (as used by `Attention`) falls out for free — and mutates each leaf in place. `SGD` is `params -= lr * grad`; `Adam` adds bias-corrected moments. `zero_grad()` sets `layer.grads = {}`.

### `basic/nn.py`

- `LayerList(list)` registers each layer as `submodules["<index>"]`.
- `Sequential` chains `forward` left-to-right and `backward` **right-to-left** — that ordering is the whole backprop mechanism; autograd-style magic isn't needed because each layer's `backward` is hand-derived.

### `basic/initializers.py` & `basic/transformers.py`

Initializers are functors (`__call__`): `Zeros/Ones/Normal/Uniform/Constant/Xavier*/Kaiming*`, used by `Linear` (default `XavierNormal`); `Constant(value, shape)` takes the value first. `transformers.py` has experimental building blocks: `Embedding` (sparse row-gradient accumulation into `grads["w"]`), `LayerNorm`, `Residual`, single-head `Attention` (hand-derived backward, nested parameter dict).

---

## Level 4 — `localdev/` and `tests/`

`localdev/` scripts aren't part of the installed package; each demos one corner:

- `nn_test.py` — **the canonical end-to-end training loop** (the exact loop from [Quick start](quickstart.md)): `Sequential([Linear(3,2), ReLU(), Linear(2,1)])` + `SGD(model.layers, 0.01)` + 1000 iterations.
- `cnn.py` — `Conv2D` forward/backward + one `SGD` step on a fixed 4×4 input.
- `test_trans.py` — `Embedding` with a hand-built table, a fixed gradient, one optimizer step; inspect `e.grads`.
- `main.py`, `temp.py` — a single-neuron mini-demo, and an Array featurure tour.

`tests/` runs with pytest (files use both `unittest` classes and pytest style) from the repo root:

```bash
python -m pytest tests/ -v        # or: .venv/bin/python -m pytest tests/
.venv/bin/python -m mypy          # type checks structs, matlib, debug (per pyproject.toml)
```

Coverage: `test_array.py` (indexing, stats, linalg, embedding-optimizer interaction, error cases), `test_debug.py` (logger, contexts, config gating, validators), `test_initializers.py` (exact fills + shapes).

---

## Suggested reading order

1. `debug/errors.py` + `debug/context.py` — 100 lines, the exception/context idiom.
2. `matlib/broadcasting.py` — how elementwise math actually works.
3. `structs/array/array.py` + `protocol.py` + `mixins.py` — the container.
4. Pick one mixin (`ops.py` is the most instructive) and the tests of it.
5. `basic/layers.py`, then `activations.py`, `criterions.py`, `optimizers.py`, `nn.py`.
6. Run `localdev/nn_test.py`; add a step counter and print `layer.grads` each iteration — that's the backprop mechanism laid bare.
