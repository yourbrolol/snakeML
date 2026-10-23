# Indexing Refactor Notes

Findings from reviewing `structs/array/indexing.py`. Items are ordered by
priority.

## High

- [x] Fix Buffer branch in `_getitem`: return `target.buffer[computed index]`,
      support negative indices; don't zip slices/lists with strides.
- [x] Add Buffer branch to `_setitem` (Buffer-backed arrays currently fail).
- [x] Enforce slice length in 1-D `_setitem` so list assignment cannot resize
      `target.data`.
- [x] Route single-key list/bool indexing (`len(key) == 1`) through recursive
      list handling instead of `target[k] = value`.
- [x] Add empty-key (0-d) guard in `_setitem` to avoid `IndexError` on
      `key[0]`.
- [x] Add key-length bound check in `_normalize_key` (too many indices /
      Ellipsis overflow).

## Medium

- [x] Decide and implement empty fancy-index behavior (`all()` on `[]` is
      `True`, so `a[[]]` mis-raises).
- [x] Raise on `zip` length mismatch for boolean/int list assignment instead of
      truncating.
- [x] Fix slice-assignment short-value fallback that passes the whole list into
      scalar slots.
- [x] Refactor `_getitem`/`_setitem` to share a selection step so single-key and
      recursive paths behave identically.
- [x] Make `_coerce_value` unwrap Buffer-valued `Array.data` (and flatten
      Buffers).

## Low

- [x] Fix Buffer branch depth check `len(key) != array.ndim` (should be
      remaining dims).
- [x] Generalize `_normalize_key` Ellipsis error message beyond `__getitem__`.
- [x] Explicitly reject bool scalar index (`a[True]`) or implement
      axis-insertion semantics.
- [x] Handle bool masks in Buffer branch (currently `bool * int` silently
      computes nonsense).
- [x] Validate int-list bounds for clearer error messages.
