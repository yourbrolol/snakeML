import builtins as _builtins

from debug import get_logger
from debug.errors import ShapeError, ValidationError
from structs.utils import zeroes, indices, set_nested

logger = get_logger(__name__)

builtins_sum = _builtins.sum



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_data(a):
    """Return the raw nested-list data from an Array or a plain list."""
    return a.data if hasattr(a, "data") else a


def _get_shape(a):
    """Return the shape tuple from an Array or compute it for a plain list."""
    if hasattr(a, "shape"):
        return a.shape

    def _shape(x):
        if not isinstance(x, list):
            return ()
        if not x:
            return (0,)
        return (len(x),) + _shape(x[0])

    return _shape(a)


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------

def concatenate(arrays, axis=0):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : list of Array
        Arrays to concatenate. All must have the same shape except along *axis*.
    axis : int, optional
        The axis along which to concatenate (default 0).

    Returns
    -------
    list
        Nested list of the concatenated result.

    Examples
    --------
    ::

        concatenate([a, b], axis=0)   # stack rows
        concatenate([a, b], axis=1)   # append columns
    """
    if not arrays:
        logger.error("concatenate called with empty sequence")
        raise ValidationError("concatenate requires at least one array")

    shapes = [_get_shape(a) for a in arrays]
    ndim = len(shapes[0])
    axis_norm = axis % ndim

    # Validate: all shapes identical except along `axis`
    ref = shapes[0]
    for s in shapes[1:]:
        if len(s) != ndim:
            logger.error("concatenate ndim mismatch", shapes=shapes)
            raise ShapeError(f"All arrays must have the same number of dimensions, got {s} vs {ref}")
        for i, (a, b) in enumerate(zip(ref, s)):
            if i != axis_norm and a != b:
                logger.error("concatenate shape mismatch", shapes=shapes, axis=axis_norm)
                raise ShapeError(
                    f"All dimensions except axis {axis_norm} must match; "
                    f"got {s} vs {ref}"
                )

    # Build output shape
    total_axis_size = builtins_sum(s[axis_norm] for s in shapes)
    out_shape = tuple(
        total_axis_size if i == axis_norm else ref[i]
        for i in range(ndim)
    )

    result = zeroes(out_shape)

    axis_offset = 0
    for arr in arrays:
        arr_shape = _get_shape(arr)
        for src_idx in indices(arr_shape):
            dst_idx = list(src_idx)
            dst_idx[axis_norm] += axis_offset
            set_nested(result, tuple(dst_idx), arr[src_idx])
        axis_offset += arr_shape[axis_norm]

    logger.debug(
        "concatenate completed",
        n_arrays=len(arrays),
        axis=axis_norm,
        out_shape=out_shape,
    )
    return result


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

def stack(arrays, axis=0):
    """Join a sequence of arrays along a *new* axis.

    Parameters
    ----------
    arrays : list of Array
        Arrays to stack. All must have the identical shape.
    axis : int, optional
        Position in the output array's shape where the new axis is inserted.

    Returns
    -------
    list
        Nested list of the stacked result.

    Examples
    --------
    ::

        stack([a, b], axis=0)   # shape (2, *a.shape)
        stack([a, b], axis=1)   # shape (a.shape[0], 2, *a.shape[1:])
    """
    if not arrays:
        logger.error("stack called with empty sequence")
        raise ValidationError("stack requires at least one array")

    shapes = [_get_shape(a) for a in arrays]
    ref = shapes[0]
    for s in shapes[1:]:
        if s != ref:
            logger.error("stack shape mismatch", shapes=shapes)
            raise ShapeError(f"All arrays must have the same shape; got {s} vs {ref}")

    n = len(arrays)
    ndim_in = len(ref)
    axis_norm = axis % (ndim_in + 1)

    # New shape: insert dimension of size n at axis_norm
    out_shape = ref[:axis_norm] + (n,) + ref[axis_norm:]
    result = zeroes(out_shape)

    for i, arr in enumerate(arrays):
        for src_idx in indices(ref):
            dst_idx = src_idx[:axis_norm] + (i,) + src_idx[axis_norm:]
            set_nested(result, dst_idx, arr[src_idx])

    logger.debug(
        "stack completed",
        n_arrays=n,
        axis=axis_norm,
        out_shape=out_shape,
    )
    return result


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split(array, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays along an axis.

    Parameters
    ----------
    array : Array
        The array to split.
    indices_or_sections : int or list of int
        * **int** — split into that many equal sections.
        * **list of int** — split points along *axis* (like numpy's behaviour).
          e.g. ``[2, 4]`` gives slices ``[:2]``, ``[2:4]``, ``[4:]``.
    axis : int, optional
        The axis along which to split (default 0).

    Returns
    -------
    list of list
        A list of nested lists, one per section.

    Raises
    ------
    ShapeError
        If an equal split is requested but the axis size is not divisible.
    ValidationError
        If ``indices_or_sections`` is 0 or negative.
    """
    shape = _get_shape(array)
    ndim = len(shape)
    axis_norm = axis % ndim
    axis_size = shape[axis_norm]

    # Build split points
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            logger.error("split sections must be positive", sections=n)
            raise ValidationError(f"sections must be a positive integer, got {n}")
        if axis_size % n != 0:
            logger.error("split does not divide evenly", axis_size=axis_size, sections=n)
            raise ShapeError(
                f"Axis {axis_norm} has size {axis_size} which is not divisible by {n}"
            )
        step = axis_size // n
        split_points = list(range(0, axis_size + 1, step))
    else:
        pts = sorted(set(indices_or_sections))
        split_points = [0] + pts + [axis_size]

    # Slice along axis for each consecutive pair of split points
    results = []
    for start, stop in zip(split_points, split_points[1:]):
        sec_size = stop - start
        sec_shape = tuple(sec_size if i == axis_norm else shape[i] for i in range(ndim))
        sec = zeroes(sec_shape)

        for sec_idx in indices(sec_shape):
            src_idx = list(sec_idx)
            src_idx[axis_norm] += start
            set_nested(sec, sec_idx, array[tuple(src_idx)])

        results.append(sec)

    logger.debug(
        "split completed",
        n_sections=len(results),
        axis=axis_norm,
        in_shape=shape,
    )
    return results


# ---------------------------------------------------------------------------
# Unbind
# ---------------------------------------------------------------------------

def unbind(array, axis=0):
    """Remove a dimension by returning all slices along it.

    Equivalent to indexing every position along *axis* and collecting the
    results. Each returned array has one fewer dimension than the input.

    Parameters
    ----------
    array : Array
        The input array.
    axis : int, optional
        The axis to unbind along (default 0).

    Returns
    -------
    list of list
        A list of ``shape[axis]`` nested lists, each with shape equal to
        the input shape with the *axis* dimension removed.

    Examples
    --------
    ::

        # a has shape (3, 4)
        slices = unbind(a, axis=0)   # → list of 3 arrays, each shape (4,)
        slices = unbind(a, axis=1)   # → list of 4 arrays, each shape (3,)
    """
    shape = _get_shape(array)
    ndim = len(shape)
    axis_norm = axis % ndim
    axis_size = shape[axis_norm]

    # Output shape: input shape with the axis dimension removed
    out_shape = tuple(s for i, s in enumerate(shape) if i != axis_norm)

    results = []
    for k in range(axis_size):
        if out_shape:
            sec = zeroes(out_shape)
            for out_idx in indices(out_shape):
                # Reconstruct the input index: insert k at axis_norm
                src_idx = out_idx[:axis_norm] + (k,) + out_idx[axis_norm:]
                set_nested(sec, out_idx, array[src_idx])
        else:
            # Scalar result (unbinding a 1-D array)
            sec = array[(k,)]
        results.append(sec)

    logger.debug(
        "unbind completed",
        axis=axis_norm,
        in_shape=shape,
        n_slices=axis_size,
        out_shape=out_shape,
    )
    return results
