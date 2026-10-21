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
