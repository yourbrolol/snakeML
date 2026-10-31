# Bufferization Roadmap

Switching `Array` from nested-list backing (`data: NestedList`) to a flat,
buffered type (`Buffer` over `list[Scalar]`). The revamp keeps the public API
1:1 identical — tests must not change behavior expectations.

Current transitional state: `Array.data` is typed `Buffer | NestedList`
(`array.py:43`), `Buffer` lives in `structs/array/buffer.py`, and
`ArrayUtilsOps._to_buffer` (`structs/array/utils.py`) already exists with
`tolist()` / `_restore_from_flat` acting as the nested ↔ flat bridges.

---

## Milestone 0 — Lock the baseline (before touching anything)

- Run `tests/test_array.py` and any other suite; record failures if any exist
  *now*. Every later milestone is gated on "tests pass, same results as M0".
- Optionally add a "mirror check": run an op on both a nested-list array and a
  buffer-backed one, assert identical results. This lets `NestedList` mode
  stay alive during migration instead of a big-bang flip.

## Milestone 1 — `Buffer` class (new file, nothing else changes)

- `structs/array/buffer.py`: flat `list[Scalar]` + invariant
  `len == prod(shape)` (shape may live on `Array` only; `Buffer` is purely
  flat storage), `__len__/__iter__/__getitem__` by flat index.
- Add two bridges in `utils.py`, reusing what is already there:
  - `_to_buffer(nested) -> Buffer` (inverse of the `_get_shape` walk) — done.
  - `tolist()` stays as `Buffer -> NestedList` via `_restore_from_flat`.
- No `Array` changes yet. Nothing can regress.
- Tests: round-trip `nested -> Buffer -> nested` identity for 0-d/1-d/3-d/
  jagged/empty cases.

## Milestone 2 — Construction path flip (`array.py` + `constructors.py`)

- `__init__` non-raw path: `_to_list(values)` → `_to_buffer(values)`;
  `raw` path requires `shape` (asserts `len(buffer) == prod(shape)`).
- `constructors.py` becomes *simpler*, not harder: `zeros/empty/full` are
  literally `Buffer([value] * prod(shape))`.
- `__len__` (`array.py:100`) → `self.shape[0]` (the
  `cast(List[Any], self.data)` hack dies here).
- `__repr__`/`_repr` → render via `_restore_from_flat`.
- Tests: full construction surface (nested lists, tuples, `range`, 0-d,
  `raw=True`).

## Milestone 3 — Indexing (`indexing.py`) — the core payoff

- `_getitem`/`_setitem` rewrite: resolve `key` → per-dim slice-or-int tuple →
  flat offset out; single-scalar, slice, mask, and fancy-index all become
  flat-offset operations on the buffer. This is the milestone where
  `Buffer |` converges to just `Buffer`.
- Decide here (it determines the rest): **copies only** (buffer always
  C-contiguous, simplest) vs. **views** (share `Buffer`, allow general
  strides). Copies first; views as a later milestone.
- Tests: existing indexing tests must pass unchanged — that is the contract.

## Milestone 4 — `utils.py` (the big walker farm)

- `_flatten`, `deepcopy`, `_fill_value`, `_masked`, `_where`, split/slice
  helpers: most nested-list recursion collapses to flat loops or slices —
  this milestone is mostly *deletion*.
- `_restore_from_flat` becomes the single shape-rebuild utility everything
  downstream uses.
- In-place mutation sites that write `self.data` then re-derive `self.shape`
  (`fill`, `put`, `shuffle`, and `ops.py:56-57`) change to "rebuild Buffer +
  set shape" — and it is worth reviewing whether those sites should stay
  in-place at all.

## Milestone 5 — `shape.py` + `ops.py` (the broadcasting/geometry layer)

- `broadcasting.elementwise` currently walks nested trees — becomes: broadcast
  shapes → flat index math → new Buffer. Biggest single rewrite of the
  roadmap.
- `shape.py` has 2-d-specific row assumptions (pad working by "rows of
  `len(self.data)`") — those get replaced by flat/stride logic that works for
  any `ndim`.

## Milestone 6 — `stats.py` + cleanup + views

- Reductions become flat loops over the buffer (easiest module once data is
  flat).
- Delete the transitional `Buffer | NestedList` union from `dtypes.py` /
  `protocol.py`; `_get_shape` gains the `len(flat) == prod(shape)` invariant
  check.
- Only now open the views door: non-zero `offset`, possibly non-contiguous
  `strides` — the feature that was blocked-in-waiting in the old design
  (`offset` was "reserved for future view support").

---

## Dependency logic

M1 isolates the primitive → M2 converts in → M3 replaces geometry → M4-M6 are
consumers that get *simpler* as you go. The union type (`Buffer | NestedList`)
exists precisely so M2-3 can land before M4-6; kill it only at M6.

## Notes / decided details

- `_to_buffer` semantics: scalars → 1-element buffer (0-d shape is
  `_get_shape`'s job); recursion bottoms out on `Buffer` instances so the
  transitional dual representation stays safe; jagged depth collapses via
  `extend`; `str`/`bytes`/`bytearray` rejected; `dict` iteration excluded
  (keys-only iteration footgun — same issue exists in `_to_list`, revisit).
- `Buffer` stores its list as `.buffer` and is deliberately not a `list`
  subclass; anything recursing over mixed `Buffer`/nested values must unwrap
  explicitly.
- Container enforcement idea for later: typed per-dtype storage
  (`array.array`/`memoryview`) can come as `Buffer.typed()` — plain
  `list[Scalar]` first, since the API currently accepts heterogeneous
  `bool`/`int`/`float`/`complex`.

---

## Post-refactor status — what is left

The `Array` main path is now bufferized: storage is `Buffer` + `View`, and
arithmetic, broadcasting, linalg, stats, manipulation, reshape/flatten, and
in-place mutations all read or write flat storage. The remaining work is
mostly bridges, dead nested fallbacks, stale annotations, invariant checks,
and directness/cleanup work. Nested bridges themselves may stay.

### Bridges that remain (acceptable, but should be explicit)

- `Array.__init__` still normalises nested input through
  `buffer_helpers._buffer_from` → `_to_buffer` + `_get_shape`.
- `tolist()` / `_unwrap` / `_restore_from_flat` rebuild nested lists from flat
  storage via `indexing._buffer_to_nested`.
- `indexing._coerce_value` converts assigned `Array`/tuple/nested values into
  nested lists before `_assign_buffer` writes them back to flat storage.
- `matlib/_array/_common.py` still accepts plain nested lists/Buffers as
  inputs and normalises them through `flat_values()`; this is the public
  input boundary, not the internal compute path.
- Keep these only at API boundaries. Remove internal callers that use a
  bridge merely to immediately flatten again.

### Indexing is buffer-backed but not yet a pure flat path

- `_getitem_buffer` resolves keys with shape/strides/offset and reads the
  `Buffer` correctly, but it materialises multi-element results as nested
  lists. `Array.__getitem__` then wraps those lists with `Array(result)`,
  which normalises and flattens them again.
- `_setitem`/`_assign_buffer` write flat offsets, but assigned sequences are
  still walked as nested values through `_broadcast_sequence` and
  `_coerce_value`.
- Dead nested branches remain in `_getitem`/`_setitem` (`target` list paths)
  and in the scalar fallback of `_scalar_value`; `Array.data` is always a
  `Buffer`, so these branches are no longer reachable from normal `Array`
  use.
- Decide whether indexing should return a flat `Buffer`/`Array` directly
  (copies only) instead of the nested-list → `Array` round trip. This is a
  directness/performance cleanup, not a current correctness bug.
- Slices currently copy; there is no view-based `__getitem__` path yet.

### Dead / transitional nested logic to remove

- `buffer_transform.fill`: remove the `isinstance(self.data, Buffer)` fallback
  and the nested `_fill_value` / `_get_shape` branch.
- `buffer_select.put` / `shuffle`: remove the `not isinstance(self.data,
  Buffer)` fallbacks; both mutation paths are Buffer-only today.
- `ops._update_self`: the non-Array fallback that calls `_get_shape(result)` is
  dead for normal buffer-backed results; simplify after indexing cleanup.
- `buffer_helpers._flatten` / `_unwrap` / `_wrap_result` / `_wrap` / `flatten`
  / `item` / `dtype` and similar standalone delegates are legacy bridges.
  Audit for remaining callers, then delete or reduce them to thin public
  adapters.
- `buffer_helpers._flatten` currently does `_Ops().data`, which both trips
  mypy (abstract instantiation) and assumes a nested-style default. Replace
  with the `Array._flatten()` path or remove it.
- `matlib/_list` remains as the legacy nested-list reference implementation.
  It is not the `Array` compute path, but its `hasattr(value, "data")`
  unwrapping assumes nested lists and can mis-handle buffer-backed values if
  called directly. Keep it out of `Array` routing, or update it explicitly.
- `matlib/_list/manipulation.py` still fails for plain nested lists because it
  indexes list objects with tuples; fix separately if the legacy list path is
  still supported.

### Construction still has nested scaffolding

- `constructors.full`, `eye`, `randn`, and `randint` build nested Python
  lists and then let `Array.__init__` flatten them. Correct, but not the
  direct `Buffer([value] * prod(shape))` construction described in M2.
- `_buffer_from` still infers shape from nested input for non-`Array`
  inputs. That is the intended bridge; only the internal constructors should
  stop generating nested scaffolding.
- Jagged/empty inputs still rely on `_get_shape` and can produce a
  shape/buffer-length mismatch that is not rejected.

### Buffer/View invariants still missing

- `View._validate()` checks rank, non-negative shape, and non-negative
  offset, but does **not** check `len(buffer) == prod(shape)`.
- `Array(raw=True, ...)` accepts any `Buffer`/shape combination without a
  length check.
- `Array.data = Buffer(...)` bypasses shape reconciliation entirely: it
  replaces the buffer and returns without updating or validating shape,
  strides, or offset. The nested-input branch does reconcile shape, so the
  two setter paths currently behave differently.
- Add a single `_validate_buffer_shape(buffer, shape)` invariant and use it
  in `View`, `raw=True`, `_from_buffer`, and the `data` setter.
- `_get_shape` should gain the roadmap invariant check once all constructors
  feed it consistent flat data.

### Stale `NestedList` annotations and mypy debt

- `structs/array/protocol.py` still declares `data: NestedList` and retains
  nested-oriented helper contracts (`_to_list`, `_get_shape`, `_unwrap`,
  `_wrap_result`, `_restore_from_flat`).
- `structs/array/utils_helpers/buffer_core.py`,
  `utils_helpers/buffer_transform.py`, and `utils_helpers/buffer_select.py`
  redeclare `data: NestedList`, causing override errors against `Array.data:
  Buffer`.
- The standalone helpers in `structs/array/buffer_helpers.py` still type
  storage/return values through `NestedList` where they should distinguish
  flat storage from public nested output.
- `ArrayProtocol` should type only the bridge-facing methods as
  `NestedList`; internal storage should be `Buffer`. Public `tolist()`
  remains `NestedList`.
- `.venv/bin/python -m mypy` currently reports **36 errors in 8 files**,
  mostly override/attr-defined/abstract-instantiation errors from these
  stale declarations and mixin cross-references. No errors were reported in
  `matlib/_array`.
- Update stale docstrings/attributes that describe `Array` as “nested-list
  backed” (`array.py` class docstring, `protocol.py` data comment, and
  related helper docs).

### Geometry / other deferred work

- `shape.pad` is still explicitly 2-D only; `roll` only implements an axis
  path for 1-D arrays. These are pre-existing functional limits, not
  bufferization regressions, but they remain outside the “works for any
  ndim” M5 goal.
- `shape.py` operations often flatten to a Python list, compute, then build a
  new `Buffer`. Correct and flat-backed, but not a stride-aware
  implementation. Optimise only if needed.
- `broadcasting.elementwise` is flat but loops element-by-element and builds
  Python lists; vectorisation/typed buffers remain future work.
- Views exist at the `View` level but are not yet exposed through indexing;
  non-contiguous views and shared-buffer slicing remain deferred.
- `Buffer` does not yet enforce its length invariant or provide typed storage;
  `Buffer.typed()` remains the planned extension.

### Tests / verification gaps

- Current verification: `.venv/bin/python -m pytest tests -q` → **31 passed**;
  `git diff --check` clean.
- Add focused coverage for: `Buffer` round-trip on 0-d/1-d/3-d/empty/jagged
  input; `raw=True` length validation; `data = Buffer(...)` shape
  reconciliation; views with non-zero `offset`/non-contiguous `strides`;
  slice/fancy/mask `__getitem__` parity against nested-list results; and
  constructor output equivalence after direct `Buffer` construction.
- Re-run `localdev` collection/tests separately from `tests`; only `tests`
  is currently verified.
