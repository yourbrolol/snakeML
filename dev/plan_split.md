# Plan: Split the largest `structs/` files

## Why

- `structs/array/utils.py` 964 LoC and `structs/array/ops.py` 909 LoC are the biggest
  files; navigating them for a single change (e.g. buffer migration) is error‑prone.
- The **Bufferization roadmap** (`dev/bufferize.md`) already carves the work into
  milestones M1‑M6.  Splitting the files along those boundaries gives each module a
  clear, small scope.
- Smaller files = faster compiles, easier reviews, and isolated bug surfaces.

## Current file sizes (approx.)

| File | LoC | Main concern |
|---|---|---|
| `structs/array/utils.py` | 964 | `_to_buffer`, `_get_shape`, `_restore_from_flat`, `_flatten`, stats helpers |
| `structs/array/ops.py` | 909 | `matmul`, `elementwise`, broadcasting, in‑place mutation, reductions |
| `structs/array/shape.py` | 554 | broadcasting helpers, tile/repeat/_broadcast, row‑assumptions |
| `structs/array/indexing.py` | 245 | `_getitem` / `_setitem` (core payoff of M3) |
| `structs/array/constructors.py` | 301 | `full`, `zeros`, `ones`, `empty`, `randn`, `randint` |

## Splitting strategy – per milestone

### Milestone 1 — `Buffer` class (already done)
- **No file split needed.** `Buffer` lives in `structs/array/buffer.py` (18 LoC).

### Milestone 2 — Construction path flip
- **`structs/array/utils.py`** – extract `_buffer_from` (new, 32 LoC) and
  `_to_buffer` / `_restore_from_flat` into a *new* small module
  `structs/array/buffer_helpers.py` (≈150 LoC).  The remaining utility functions
  (`_get_shape`, `_strides`, `_flatten`, `copy`/`clone`) stay in `utils.py` but
  are marked “transition” – they will be deleted in later milestones.
- **`structs/array/array.py`** – the `__init__` change (non‑raw path uses
  `_buffer_from`) is the only code touched here; everything else stays untouched.

### Milestone 3 — Indexing (`indexing.py`)
- **`structs/array/indexing.py`** becomes the *only* place that understands both
  `Buffer` and nested‑list representations.
- The file is already modest (245 LoC).  After M3 the `_getitem`/`_setitem`
  implementation can be rewritten to work on the flat buffer; the nested‑list
  fallback (`_nested()` shim) can be removed.
- **No new file**; just refactor the existing module.

### Milestone 4 — `utils.py` walker farm
- **Split `structs/array/utils.py`** into:
  - `structs/array/buffer_helpers.py` – `_to_buffer`, `_restore_from_flat`,
    `_buffer_from`, plus any flat‑only helpers (`_fill_value`, `_masked`,
    `_where`).
  - `structs/array/stats_helpers.py` – reduction functions that now operate on
    flat buffers (milestone 5 will move them here).
  - The remaining `utils.py` shrinks to <200 LoC, containing only what’s needed
    by the transition shim (`_nested`, etc.).

### Milestone 5 — `shape.py` + `ops.py` broadcasting/geometry
- **`structs/array/shape.py`** (554 LoC) → two modules:
  - `structs/array/broadcasting.py` – the `broadcasting.elementwise` logic and
    any shape‑inference helpers.  This is the “biggest single rewrite” per the
    roadmap; extracting it leaves `shape.py` with only low‑level stride math.
  - `structs/array/shape_core.py` – strides, padding, tiling, repeating that
    were originally 2‑d‑specific and now get replaced by flat/stride logic.
- **`structs/array/ops.py`** (909 LoC) → three modules:
  - `structs/array/reductions.py` – all reduction (`sum`, `mean`, `var`, etc.)
    moved out per milestone 6.
  - `structs/array/broadcasting.py` – element‑wise ops, broadcasting
    (extracted from the current `ops.py`; overlaps with the new
    `structs/array/broadcasting.py` but keeps the API surface).
  - `structs/array/ops_core.py` – the remaining `matmul`, `outer`, `tile`,
    `repeat` that still work on either representation.

### Milestone 6 — `stats.py` + cleanup + views
- **`structs/array/stats.py`** (279 LoC) – reductions become flat loops over the
  buffer; the file can be collapsed into `structs/array/reductions.py` (from
  M5 split) or kept tiny.
- **`structs/dtypes.py` / `protocol.py`** – remove the `Buffer | NestedList`
  union; the type becomes just `Buffer` after M5‑M6 are complete.
- **Views** – `offset` and non‑contiguous `strides` can now be opened because
  the flat representation is proven.

## Concrete next steps (what to do today)

1. **Extract `_buffer_from`** – already committed (`216f7b7`).  It lives in
   `structs/array/utils.py` but is deliberately isolated; you can move it to
   `structs/array/buffer_helpers.py` whenever you’re ready.
2. **Add a `_nested` property** to `Array` (see the earlier answer) so that all
   15 downstream `self.data` reads use `self._nested()` instead.  This keeps
   tests green while the underlying storage is already flat.
3. **Rename / split `utils.py`** per the table above – a simple `mv` + import
   updates in the call sites.  Because `_buffer_from` is the only new public
   surface, the rename is mechanically safe.
4. **Update `ops.py` imports** to reference the new module names; the roadmap
   guarantees that Milestone 5 work won’t break M2‑M3.

## Checklist after each milestone

| Milestone | Files that may change | Tests that must pass |
|---|---|---|
| M1 | `buffer.py` (added) | all existing + round‑trip nested↔Buffer |
| M2 | `array.py` `__init__`, `utils.py` `_buffer_from` | `tests/test_array.py` (all 9) |
| M3 | `indexing.py` `_getitem`/`_setitem` | same 9 tests + indexing stress tests |
| M4 | `utils.py` → helpers, new modules | same 9 + construction-surface tests |
| M5 | `shape.py` → `broadcasting.py` + `shape_core.py`; `ops.py` → splits | same 9 + broadcasting correctness |
| M6 | `stats.py`, `dtypes.py`, union removal | full test suite, view‑related tests |

## How to keep the split minimal

- **One file at a time.**  Pick the smallest milestone (M2 is just the
  `_buffer_from` addition) and commit before touching the next.
- **Import‑only changes.**  When you move a function, update *all* call sites
  in a single `git grep -l "from structs.array.utils import"` pass; the function
  signature stays identical.
- **No behavioural change.**  Each split preserves the exact same Python
  object/model; only the *location* of the code moves.  Run the full test
  suite after every commit.

---

**Result:** After M6 the `structs/array/` package will be a collection of focused
modules (≈150 LoC each) instead of two 900‑plus‑line monstrosities, and the
bufferisation roadmap will have been traversed step‑by‑step with green tests
throughout.