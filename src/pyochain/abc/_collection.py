from __future__ import annotations

import itertools
from abc import ABC
from collections.abc import Collection
from typing import TYPE_CHECKING, Self, override

from ._iterable import PyoIterable

if TYPE_CHECKING:
    from .._iter import Iter


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

    def repeat(self, n: int | None = None) -> Iter[Self]:
        """Repeat the entire `Collection` **n** times (as elements) in an `Iter`.

        If **n** is `None`, repeat indefinitely.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            [`Iter::cycle`][cycle] to repeat the *elements* of the `Iter` indefinitely.

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[Self]: An `Iter` of repeated `Iter`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2)).repeat(3).collect()
            Seq(Seq(1, 2), Seq(1, 2), Seq(1, 2))
            >>> Seq(("a", "b")).repeat(2).collect()
            Seq(Seq('a', 'b'), Seq('a', 'b'))
            >>> Seq([0]).repeat().flatten().take(5).collect()
            Seq(0, 0, 0, 0, 0)

            ```
        """
        from .._iter import Iter

        if n is None:
            return Iter(itertools.repeat(self))
        return Iter(itertools.repeat(self, n))

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
