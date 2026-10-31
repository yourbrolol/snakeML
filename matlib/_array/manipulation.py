"""Array-native manipulation kernels: concatenate, stack, split, flip, unbind."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

from debug import get_logger
from debug.errors import ShapeError, ValidationError

from ._common import (
    array_from,
    array_view,
    coords_from_flat,
    dropped_axis_view,
    flat_from_coords,
    flat_values,
    shape_of,
    size_of,
    sliced_view,
)

logger = get_logger(__name__)

__all__ = ["concatenate", "flip", "stack", "split", "unbind"]


def concatenate(arrays: Sequence[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along an existing axis.

    Raises:
        ValidationError: If ``arrays`` is empty.
        ShapeError: If ranks differ or a non-``axis`` dimension mismatches.
    """
    if not arrays:
        logger.error("concatenate called with empty sequence")
        raise ValidationError("concatenate requires at least one array")

    shapes = [flat_values(array)[1] for array in arrays]
    ndim = len(shapes[0])
    axis_norm = axis % ndim

    reference = shapes[0]
    for candidate in shapes[1:]:
        if len(candidate) != ndim:
            logger.error("concatenate ndim mismatch", shapes=shapes)
            raise ShapeError(
                "All arrays must have the same number of dimensions, "
                f"got {candidate} vs {reference}"
            )
        for index, (left, right) in enumerate(zip(reference, candidate)):
            if index != axis_norm and left != right:
                logger.error(
                    "concatenate shape mismatch", shapes=shapes, axis=axis_norm
                )
                raise ShapeError(
                    f"All dimensions except axis {axis_norm} must match; "
                    f"got {candidate} vs {reference}"
                )

    total_axis_size = sum(candidate[axis_norm] for candidate in shapes)
    out_shape = tuple(
        total_axis_size if index == axis_norm else reference[index]
        for index in range(ndim)
    )

    result: List[Any] = [None] * size_of(out_shape)
    axis_offset = 0
    for array in arrays:
        values, shape = flat_values(array)
        for src_index in range(size_of(shape)):
            coords = coords_from_flat(src_index, shape)
            dst_coords = list(coords)
            dst_coords[axis_norm] += axis_offset
            result[flat_from_coords(dst_coords, out_shape)] = values[src_index]
        axis_offset += shape[axis_norm]

    logger.debug(
        "concatenate completed",
        n_arrays=len(arrays),
        axis=axis_norm,
        out_shape=out_shape,
    )
    return array_from(result, out_shape)


def flip(array: Any, axis: int) -> Any:
    """Flip an array along the specified axis.

    ``Array`` inputs return a negative-stride view sharing the original
    Buffer; other inputs fall back to a materialised contiguous result.
    """
    shape = shape_of(array)
    ndim = len(shape)
    axis_norm = axis % ndim

    strides = getattr(array, "strides", None)
    if strides is not None:
        new_strides = list(strides)
        new_strides[axis_norm] = -new_strides[axis_norm]
        offset = getattr(array, "offset", 0)
        if shape[axis_norm]:
            offset += (shape[axis_norm] - 1) * strides[axis_norm]
        view = array_view(array, shape, new_strides, offset)
        if view is not None:
            logger.debug("flip completed as view", axis=axis_norm, shape=shape)
            return view

    values, shape = flat_values(array)
    result = [None] * size_of(shape)
    for index in range(size_of(shape)):
        coords = coords_from_flat(index, shape)
        src_coords = list(coords)
        src_coords[axis_norm] = shape[axis_norm] - 1 - coords[axis_norm]
        result[index] = values[flat_from_coords(src_coords, shape)]

    logger.debug("flip completed", axis=axis_norm, in_shape=shape, out_shape=shape)
    return array_from(result, shape)


def stack(arrays: Sequence[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along a new axis.

    Raises:
        ValidationError: If ``arrays`` is empty.
        ShapeError: If the arrays' shapes are not identical.
    """
    if not arrays:
        logger.error("stack called with empty sequence")
        raise ValidationError("stack requires at least one array")

    shapes = [flat_values(array)[1] for array in arrays]
    reference = shapes[0]
    for candidate in shapes[1:]:
        if candidate != reference:
            logger.error("stack shape mismatch", shapes=shapes)
            raise ShapeError(
                f"All arrays must have the same shape; got {candidate} vs {reference}"
            )

    count = len(arrays)
    ndim_in = len(reference)
    axis_norm = axis % (ndim_in + 1)
    out_shape = reference[:axis_norm] + (count,) + reference[axis_norm:]

    result: List[Any] = [None] * size_of(out_shape)
    for position, array in enumerate(arrays):
        values, shape = flat_values(array)
        for src_index in range(size_of(shape)):
            coords = coords_from_flat(src_index, shape)
            dst_coords = coords[:axis_norm] + [position] + coords[axis_norm:]
            result[flat_from_coords(dst_coords, out_shape)] = values[src_index]

    logger.debug(
        "stack completed",
        n_arrays=count,
        axis=axis_norm,
        out_shape=out_shape,
    )
    return array_from(result, out_shape)


def split(
    array: Any,
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0,
) -> List[Any]:
    """Split an array into multiple sub-arrays along an axis.

    Raises:
        ShapeError: If an equal split is requested but the axis size is not
            divisible.
        ValidationError: If ``indices_or_sections`` is 0 or negative.
    """
    shape = shape_of(array)
    ndim = len(shape)
    axis_norm = axis % ndim
    axis_size = shape[axis_norm]

    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            logger.error("split sections must be positive", sections=sections)
            raise ValidationError(
                f"sections must be a positive integer, got {sections}"
            )
        if axis_size % sections != 0:
            logger.error(
                "split does not divide evenly", axis_size=axis_size, sections=sections
            )
            raise ShapeError(
                f"Axis {axis_norm} has size {axis_size} which is not divisible "
                f"by {sections}"
            )
        step = axis_size // sections
        split_points = list(range(0, axis_size + 1, step))
    else:
        points = sorted(set(indices_or_sections))
        split_points = [0] + points + [axis_size]

    views: List[Any] = []
    for start, stop in zip(split_points, split_points[1:]):
        view = sliced_view(array, axis_norm, start, stop)
        if view is None:
            break
        views.append(view)
    if len(views) == len(split_points) - 1:
        logger.debug(
            "split completed as views",
            n_sections=len(views),
            axis=axis_norm,
            in_shape=shape,
        )
        return views

    values, shape = flat_values(array)
    results: List[Any] = []
    for start, stop in zip(split_points, split_points[1:]):
        section_size = stop - start
        section_shape = tuple(
            section_size if index == axis_norm else shape[index]
            for index in range(ndim)
        )
        section: List[Any] = [None] * size_of(section_shape)
        for section_index in range(size_of(section_shape)):
            coords = coords_from_flat(section_index, section_shape)
            src_coords = list(coords)
            src_coords[axis_norm] += start
            section[section_index] = values[flat_from_coords(src_coords, shape)]
        results.append(array_from(section, section_shape))

    logger.debug(
        "split completed",
        n_sections=len(results),
        axis=axis_norm,
        in_shape=shape,
    )
    return results


def unbind(array: Any, axis: int = 0) -> List[Any]:
    """Remove a dimension by returning all slices along it.

    Multi-dimensional ``Array`` inputs return axis-dropped views sharing the
    original Buffer; other inputs fall back to materialised results.
    """
    shape = shape_of(array)
    ndim = len(shape)
    axis_norm = axis % ndim
    axis_size = shape[axis_norm]

    out_shape = tuple(
        dim for index, dim in enumerate(shape) if index != axis_norm
    )

    if out_shape:
        views: List[Any] = []
        for position in range(axis_size):
            view = dropped_axis_view(array, axis_norm, position)
            if view is None:
                break
            views.append(view)
        if len(views) == axis_size:
            logger.debug(
                "unbind completed as views",
                axis=axis_norm,
                in_shape=shape,
                n_slices=axis_size,
                out_shape=out_shape,
            )
            return views

    values, shape = flat_values(array)
    results: List[Any] = []
    for position in range(axis_size):
        if out_shape:
            section: List[Any] = [None] * size_of(out_shape)
            for out_index in range(size_of(out_shape)):
                coords = coords_from_flat(out_index, out_shape)
                src_coords = coords[:axis_norm] + [position] + coords[axis_norm:]
                section[out_index] = values[flat_from_coords(src_coords, shape)]
            results.append(array_from(section, out_shape))
        else:
            results.append(values[position])

    logger.debug(
        "unbind completed",
        axis=axis_norm,
        in_shape=shape,
        n_slices=axis_size,
        out_shape=out_shape,
    )
    return results
