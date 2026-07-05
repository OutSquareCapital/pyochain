from __future__ import annotations

from collections.abc import MutableSet

from ._iterator import (  # pyright: ignore[reportMissingModuleSource]
    PyoSet,
)


class PyoMutableSet[T](PyoSet[T], MutableSet[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """ABCs for read-only and mutable sets."""

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]
