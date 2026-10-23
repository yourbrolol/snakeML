from __future__ import annotations

from typing import Optional, Union, cast

from structs.dtypes import Scalar


class Buffer:
    """A flat, 1-D row-major container of scalars backing :class:`Array`.

    Attributes:
        buffer: The underlying ``list[Scalar]`` storage.
    """

    buffer: list[Scalar]

    def __init__(self, buffer: Optional[list[Scalar]] = None):
        self.buffer = buffer if buffer is not None else []

    def __len__(self) -> int:
        """Number of scalar elements in the buffer.

        Returns:
            int: The length of the underlying storage.
        """
        return len(self.buffer)

    def __getitem__(self, key: Union[int, slice]) -> Union[Scalar, list[Scalar]]:
        """Read one element or a slice of the flat storage.

        Args:
            key: An ``int`` position or a ``slice``.

        Returns:
            Scalar | list[Scalar]: The scalar at ``key``, or the sliced list.

        Example:
            >>> Buffer([1, 2, 3])[1]
            2
        """
        if isinstance(key, slice):
            return self.buffer[key]
        return self.buffer[key]

    def __setitem__(self, key: Union[int, slice], value: Union[Scalar, list[Scalar]]) -> None:
        """Write one element or a slice of the flat storage.

        Args:
            key: An ``int`` position or a ``slice``.
            value: A scalar for a single position, or a list of scalars for a
                slice.

        Example:
            >>> buf = Buffer([1, 2, 3])
            >>> buf[1] = 9
            >>> buf.buffer
            [1, 9, 3]
        """
        if isinstance(key, slice):
            self.buffer[key] = cast("list[Scalar]", value)
        else:
            self.buffer[key] = cast("Scalar", value)
