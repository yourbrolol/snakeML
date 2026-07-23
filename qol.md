# QoL improvement plan for SnakeML

## Goals
- Make the library easier to learn and use for newcomers.
- Reduce friction in the common training workflow.
- Improve maintainability without changing the core design too aggressively.

## Phase 1: usability fixes
- Add defaults for common constructors such as `Linear` and `Sequential`.
- Improve import ergonomics by exporting the most-used symbols from package entry points.
- Add a short getting-started example and a minimal training example to the README.

## Phase 2: reliability and developer experience
- Add regression tests for constructor defaults, simple forward/backward passes, and optimizer updates.
- Standardize error messages for shape and type validation.
- Make debug logging less noisy while preserving useful context.

## Phase 3: maintainability and packaging
- Add packaging metadata so the project installs cleanly with `pip install -e .`.
- Add a basic CI workflow for automated tests.
- Document the intended architecture and the public API surface.

## Immediate priorities
1. Fix the current constructor ergonomics issue that blocks the local development example.
2. Ensure the core training flow works with minimal code.
3. Add a small set of regression tests around the most common user path.

## Observed bottlenecks
- The current `Linear` API requires an initializer explicitly, which makes even simple examples awkward and is a clear usability blocker.
- `Array` construction and elementwise math repeatedly rebuild nested Python lists, which is expensive for larger tensors and training loops.
- The tensor operations in `structs/linalg.py` and `structs/broadcasting.py` are implemented with nested Python loops and recursion rather than vectorized operations, so they will scale poorly.
- Debug logging is emitted throughout the forward/backward path, which adds overhead to every training step and can dominate small workloads.
- The convolution path in `basic/layers.py` relies on repeated patch materialization and shape manipulation, so it is likely to be much slower than a more specialized implementation.
