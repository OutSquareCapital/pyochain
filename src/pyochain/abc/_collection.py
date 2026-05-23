from __future__ import annotations

from abc import ABC
from collections.abc import Collection
from typing import override

from ._iterable import PyoIterable


class PyoCollection[T](PyoIterable[T], Collection[T], ABC):
    """`Extends `PyoIterable[T]` and `collections.abc.Collection[T]`.

    This includes `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, etc...

    Any concrete subclass must implement the required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def length(self) -> int:
        return len(self)

    @override
    def all_unique(self) -> bool:
        return len(self) == len(frozenset(self))

    def contains(self, value: T) -> bool:
        """Check if the `Collection` contains the specified **value**.

        This is equivalent to using the `in` keyword directly on the `Collection`.

        Args:
            value (T): The value to check for existence.

        Returns:
            bool: True if the value exists in the Collection, False otherwise.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict.from_ref({1: "a", 2: "b"})
            >>> data.contains(1)
            True
            >>> data.contains(3)
            False

            ```
        """
        return value in self

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
