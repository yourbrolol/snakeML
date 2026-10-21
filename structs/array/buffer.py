from __future__ import annotations

from typing import Optional

from structs.dtypes import Scalar


class Buffer:
    """A flat, 1-D row-major container of scalars backing :class:`Array`.

    Attributes:
        buffer: The underlying ``list[Scalar]`` storage.
    """

    buffer: list[Scalar]

    def __init__(self, buffer: Optional[list[Scalar]] = None):
        self.buffer = buffer if buffer is not None else []
