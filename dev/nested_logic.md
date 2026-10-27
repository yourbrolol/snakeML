# Array buffer refactor plan

Goal: `Array` internals operate on raw `Buffer`s. Nested lists are allowed only at
public API boundaries (input conversion, output conversion, display).

Rule: only refactor `Array` and the functions on its operational dependency path.
Everything else keeps its nested logic and is parked at the bottom.

## Refactor path (in order)

### P0 — Boundary contract

- `Array.__init__` — `structs/array/array.py:50` — stop storing nested lists; always
  build/store a `Buffer` via `_buffer_from`.
- `_buffer_from` — `structs/array/buffer_helpers.py:76`
- `_to_buffer` — `structs/array/buffer_helpers.py:17`
- `_get_shape` — `structs/array/buffer_helpers.py:47` — reads nested input only.
- `Buffer` — `structs/array/buffer.py:8` — ensure the operations below can use it
  directly (iteration, slicing, in-place writes).

These three helpers remain the conversion boundary. They may keep nested-list logic.

### P1 — Core Array data path

- `Array.__len__`, `__iter__`, `__getitem__`, `__setitem__`
- `indexing._getitem` — `structs/array/indexing.py:314`
- `indexing._setitem` — `structs/array/indexing.py:466`
- `indexing._getitem_buffer` — `structs/array/indexing.py:269`
- `indexing._assign_buffer` — `structs/array/indexing.py:420`
- `indexing._coerce_value` — `structs/array/indexing.py:242`
- `indexing._scalar_value` — `structs/array/indexing.py:106`
- `indexing._broadcast_sequence` — `structs/array/indexing.py:119`
- `indexing._value_ndim` — `structs/array/indexing.py:97`

Once every target is a `Buffer`, the nested branches in `_getitem`/`_setitem` and the
value-shape helpers can be reduced to boundary-only code.

### P2 — Array utility operations

- `_flatten` — `structs/array/utils.py:197`
- `flatten` — `structs/array/utils.py:213`
- `dtype.infer` — `structs/array/utils.py:267`
- `_cast` (`astype`) — `structs/array/utils.py:507`
- `_fill_value` (`fill`) — `structs/array/utils.py:858`
- `_where` (`where`) — `structs/array/utils.py:552`
- `_masked` (`masked_fill`) — `structs/array/utils.py:651`
- `_slice_along_axis` — `structs/array/utils.py:801`
- `array_split` — `structs/array/utils.py:438`
- `hstack` — `structs/array/utils.py:410`
- `take` — `structs/array/utils.py:562`
- `put` — `structs/array/utils.py:583`
- `shuffle` — `structs/array/utils.py:725`
- `permutation` — `structs/array/utils.py:739`
- `_unwrap` — `structs/array/utils.py:135`
- `copy` — `structs/array/utils.py:118`

### P3 — Shape operations

- `reshape.build` — `structs/array/shape.py:109`
- `broadcast_to._broadcast` — `structs/array/shape.py:371`
- `repeat._repeat` — `structs/array/shape.py:433`
- `tile._tile` — `structs/array/shape.py:468`
- `pad` — `structs/array/shape.py:514`

Then verify the delegating shape methods consume `Buffer` correctly: `flip`,
`squeeze`, `unsqueeze`, `swapaxes`, `moveaxis`, `T`, `permute`, `transpose`,
`expand_dims`, `roll`.

### P4 — Math/stat kernels on the Array path

- `ArrayMathOps._elementwise` — `structs/array/ops.py:24`
- `broadcast` — `matlib/broadcasting.py:15`
- `elementwise` — `matlib/broadcasting.py:80`
- `_apply_unary` — `matlib/ops.py:44`
- `linalg.matmul` — `matlib/linalg.py:355`
- `linalg.outer` — `matlib/linalg.py:406`
- `linalg.matvec` — `matlib/linalg.py:327`
- `linalg.det` — `matlib/linalg.py:35`
- `linalg.transpose` — `matlib/linalg.py:182`
- `linalg.inverse` — `matlib/linalg.py:72`
- `linalg.solve` — `matlib/linalg.py:138`
- `linalg.cross` — `matlib/linalg.py:105`
- `linalg.cholesky` — `matlib/linalg.py:440`
- `linalg.permute` — `matlib/linalg.py:207`
- `linalg.tensordot` — `matlib/linalg.py:245`
- `manipulation.concatenate` — `matlib/manipulation.py:55`
- `manipulation.flip` — `matlib/manipulation.py:134`
- `manipulation.stack` — `matlib/manipulation.py:181`
- `manipulation.split` — `matlib/manipulation.py:245`
- `manipulation.unbind` — `matlib/manipulation.py:327`
- `stats._reduce_axis` and the `*_axis` wrappers — `matlib/stats.py:201`

### P5 — Retire list primitives (last)

Only once P1–P4 no longer call them:

- `zeroes` — `structs/utils.py:74`
- `set_nested` — `structs/utils.py:95`
- `indices` — `structs/utils.py:50`
- `build_index` — `structs/utils.py:20`
- `newarr` — `structs/utils.py:132` — delete or deprecate.

## Leave as-is (bottom — not part of this refactor)

Converters, frontend factories, display, and legacy list-only paths:

- `_to_list` — `structs/array/utils.py:42` — input normalizer; tag deprecated.
- `tolist` — `structs/array/utils.py:289` — nested output converter.
- `_restore_from_flat` — `structs/array/buffer_helpers.py:59` — output/legacy converter.
- `_buffer_to_nested` — `structs/array/indexing.py:224` — nested boundary only.
- `manipulation._get_shape._shape` — `matlib/manipulation.py:39` — plain-list fallback.
- `full.build` — `structs/array/constructors.py:53` — nested factory.
- `eye` — `structs/array/constructors.py:84` — nested factory.
- `randn.build` — `structs/array/constructors.py:155` — nested factory.
- `randint.build` — `structs/array/constructors.py:187` — nested factory.
- `Array._repr` — `structs/array/array.py:155` — display only.
- `debug/formatter.py:67 describe` — outer length only.
- `debug/diagnostics.py:16 collect_diagnostics` — passes data through.
- `matlib/rng.py`, `matlib/operators.py`, `matlib/reductions.py` — no nested logic.

## Done when

- `Array.data` is a `Buffer` after every construction path.
- Indexing, utility, shape, math and stat operations never traverse nested lists.
- Only converters/factories/display still see nested lists.
