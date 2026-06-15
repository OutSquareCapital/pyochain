from __future__ import annotations

from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import Any, override

from .._set import Set, SetMut
from .._utils import get_repr
from ..abc import PyoMutableSet


class StableSet[T](PyoMutableSet[T]):  # noqa: PLW1641
    """A mutable collection of unique elements which remember their insertion order.

    Uses a `dict` as the underlying data structure to maintain insertion order while ensuring uniqueness of elements.

    Thus, it has the same characteristics of "standard" sets, with lookup and iteration speed the same as a `dict`.

    This is very similar to using `Dict::from_keys` with `None` values, but with a specialized interface for set operations.

    Note:
        This is not the same as `sortedcontainers`, i.e it does not maintain the elements in sorted order, but rather in the order they were inserted.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the set with.

    Examples:
        ```python
        >>> from pyochain.collections import StableSet
        >>>
        >>> s = StableSet(("a", "b", "c"))
        >>> s
        StableSet('a', 'b', 'c')
        >>> s.add("d")
        >>> s
        StableSet('a', 'b', 'c', 'd')
        >>> s.discard("b")
        >>> s
        StableSet('a', 'c', 'd')

        ```
    """

    _inner: dict[T, None]
    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = dict.fromkeys(data)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner.keys())})"

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._inner

    @override
    def __eq__(self, other: object) -> bool:
        match other:
            case Set() | SetMut():
                return self._inner.keys() == other.inner  # pyright: ignore[reportUnknownMemberType]
            case AbstractSet():
                return self._inner.keys() == other
            case _:
                return False

    @staticmethod
    def from_ref[V](data: dict[V, Any]) -> StableSet[V]:  # pyright: ignore[reportExplicitAny]
        """Create a `StableSet` from a reference to an existing `dict`.

        This method wraps the provided `dict` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `StableSet` from foreign functions that return `dict` objects.

        Warning:
            Since the `StableSet` directly references the original `dict`, any modifications made to the `StableSet` will also affect the original `dict`, and vice versa.

        Args:
            data (dict[V, Any]): The `dict` to wrap.

        Returns:
            StableSet[V]: A new `StableSet` instance.

        Example:
            ```python
            >>> from pyochain.collections import StableSet
            >>>
            >>> original = {"Alice": 30, "Bob": 25, "Charlie": 35}
            >>> set_obj = StableSet.from_ref(original)
            >>> set_obj
            StableSet('Alice', 'Bob', 'Charlie')
            >>> original["David"] = 40
            >>> set_obj
            StableSet('Alice', 'Bob', 'Charlie', 'David')

            ```
        """
        instance: StableSet[V] = StableSet.__new__(StableSet)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @override
    def add(self, value: T) -> None:
        self._inner[value] = None

    @override
    def discard(self, value: T) -> None:
        del self._inner[value]

    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[T]:  # pyright: ignore[reportExplicitAny]
        return SetMut.from_ref(self._inner.keys() & other)

    @override
    def union(self, other: Iterable[T]) -> SetMut[T]:
        return SetMut.from_ref(self._inner.keys() | other)

    @override
    def difference(self, other: Iterable[Any]) -> SetMut[T]:  # pyright: ignore[reportExplicitAny]
        return SetMut.from_ref(self._inner.keys() - other)

    @override
    def symmetric_difference(self, other: Iterable[T]) -> SetMut[T]:
        return SetMut.from_ref(self._inner.keys() ^ other)
