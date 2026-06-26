from __future__ import annotations

from abc import ABC
from collections.abc import Collection, Sized

from ._iterator import (  # pyright: ignore[reportMissingModuleSource]
    PyoContainer,
    PyoIterable,
)


class PyoSized(Sized, ABC):
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def len(self) -> int:
        """Return the length of `Self`.

        Equivalent to `len(self)`, but as a method.

        Returns:
            int: The number of elements in `Self`.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict.from_ref({1: "a", 2: "b"})
            >>> data.len()
            2

            ```
        """
        return len(self)

    def is_empty(self) -> bool:
        """Returns `True` if the `Collection` contains no elements.

        Returns:
            bool: `True` if the `Collection` is empty, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d = Dict(())
            >>> d.is_empty()
            True
            >>> d.insert(1, "a")
            NONE
            >>> d.is_empty()
            False

            ```
        """
        return len(self) == 0


class PyoCollection[T](PyoIterable[T], PyoContainer[T], PyoSized, Collection[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """`Extends `PyoIterable[T]` and `collections.abc.Collection[T]`.

    This includes `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, etc...

    Any concrete subclass must implement the required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]
