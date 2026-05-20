from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    MutableSet,
)
from typing import override

from ._utils import get_repr
from .traits import PyoSet


class Set[T](PyoSet[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `Collection` Protocol from `collections.abc`, so it can be used as a standard immutable collection.

    The underlying data structure is a `frozenset`.

    Tip:
        - `Set(frozenset)` is a no-copy operation since Python optimizes this under the hood.
        - If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
            data (Iterable[T]): The data to initialize the Set with.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    __match_args__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]
    _inner: frozenset[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = frozenset(data)

    @property
    def inner(self) -> frozenset[T]:
        """Get the underlying `frozenset` data structure.

        Useful when interoperating with functions that require a standard Python `frozenset`.

        Returns:
            frozenset[T]: The underlying frozenset.
        """
        return self._inner

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner)})"

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._inner

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)


class SetMut[T](Set[T], MutableSet[T]):
    """A mutable `set` wrapper with functional API.

    Unlike `Set` which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    Tip:
        If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
        data (Iterable[T]): The mutable set to wrap.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: set[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = set(data)  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    @override
    def inner(self) -> set[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Get the underlying `set` data structure.

        Useful when interoperating with functions that require a standard Python `set`.

        Returns:
            set[T]: The underlying set.
        """
        return self._inner

    @staticmethod
    def from_ref[V](data: set[V]) -> SetMut[V]:
        """Create a `SetMut` from a reference to an existing `set`.

        This method wraps the provided `set` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `SetMut` from foreign functions that return `set` objects.

        Warning:
            Since the `SetMut` directly references the original `set`, any modifications made to the `SetMut` will also affect the original `set`, and vice versa.

        Args:
            data (set[V]): The `set` to wrap.

        Returns:
            SetMut[V]: A new `SetMut` instance wrapping the provided `set`.

        Example:
        ```python
        >>> from pyochain import SetMut
        >>> original_set = {1, 2, 3}
        >>> set_obj = SetMut.from_ref(original_set)
        >>> set_obj
        SetMut(1, 2, 3)
        >>> original_set.add(4)
        >>> set_obj
        SetMut(1, 2, 3, 4)


        ```
        """
        instance: SetMut[V] = SetMut.__new__(SetMut)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @override
    def add(self, value: T) -> None:
        """Add an element to **self**.

        Args:
            value (T): The element to add.

        Example:
        ```python
        >>> from pyochain import SetMut
        >>> s = SetMut(("a", "b"))
        >>> s.add("c")
        >>> s.iter().sort()
        Vec('a', 'b', 'c')

        ```
        """
        self._inner.add(value)

    @override
    def discard(self, value: T) -> None:
        """Remove an element from **self** if it is a member.

        Unlike `.remove()`, the `discard()` method does not raise an exception when an element is missing from the set.

        Args:
            value (T): The element to remove.

        Example:
        ```python
        >>> from pyochain import SetMut
        >>> s = SetMut(("a", "b", "c"))
        >>> s.discard("b")
        >>> s.iter().sort()
        Vec('a', 'c')

        ```
        """
        self._inner.discard(value)
