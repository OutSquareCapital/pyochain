from __future__ import annotations

from collections.abc import Collection

from ._iterator import (  # pyright: ignore[reportMissingModuleSource]
    PyoContainer,
    PyoIterable,
    PyoSized,
)


class PyoCollection[T](PyoIterable[T], PyoContainer[T], PyoSized, Collection[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """`Extends `PyoIterable[T]` and `collections.abc.Collection[T]`.

    This includes `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, etc...

    Any concrete subclass must implement the required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]
