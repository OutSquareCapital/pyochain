from __future__ import annotations

from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import Any, Final, Self, override

from ._utils import get_repr
from .abc import PyoMutableSet, PyoSet

# TODO: address the following note from official python docs regarding Set performance, with benchmarks:
# To override the comparisons (presumably for speed, as the semantics are fixed),
# redefine __le__() and __ge__(), then the other operations will automatically follow suit.


class Set[T](PyoSet[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `collections::abc::Collection` Protocol, so it can be used as a standard immutable collection.

    The underlying data structure is a `frozenset`.

    Tip:
        - `Set(frozenset)` is a no-copy operation since Python optimizes this under the hood.
        - If you have an existing `set`, consider using [`SetMut::from_ref`][SetMut.from_ref] to avoid unnecessary copying.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the set with.

    Example:
        ```python
        >>> from pyochain import Set
        >>> Set(())
        Set()
        >>> s = Set((1, 2, 2, 3))
        >>> s
        Set(1, 2, 3)
        >>> s_2 = Set(s.inner)
        >>> # No copy is made when creating s_2 from s.inner, they reference the same underlying frozenset.
        >>> is_no_copy = (
        ...     s.inner is s_2.inner
        ...     and s.inner is s.inner
        ...     and s_2.inner is s.inner
        ...     and frozenset(s.inner) is s.inner
        ... )
        >>> is_no_copy
        True
        >>> # However, creating a new Set from s (not using .inner) will be a copy operation.
        >>> Set(s).inner is s.inner
        False

        ```
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: Final[frozenset[T]]

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

    @override
    def __eq__(self, other: object) -> bool:
        return _set_eq(self, other)

    @override
    def __hash__(self) -> int:
        return hash(self._inner)

    @override
    def intersection(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner & other)

    @override
    def union(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner | other)

    @override
    def difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner - other)

    @override
    def symmetric_difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner ^ other)


class SetMut[T](PyoMutableSet[T]):  # noqa: PLW1641
    """A mutable, unordered collection of unique elements.

    Unlike [`Set`][Set] which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `collections::abc::MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    Tip:
        If you have an existing `set`, consider using [`SetMut::from_ref`][from_ref] to avoid unnecessary copying.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the set with.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: set[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = set(data)

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
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner)})"

    @override
    def __eq__(self, other: object) -> bool:
        return _set_eq(self, other)

    @property
    def inner(self) -> set[T]:
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
            SetMut[V]: A new `SetMut` instance.

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

        Unlike [`SetMut::remove`][remove], this method does not raise an exception when an element is missing from the set.

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

    @override
    def intersection(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner & other)

    @override
    def union(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner | other)

    @override
    def difference(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner - other)

    @override
    def symmetric_difference(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner ^ other)


class SetMutOrdered[T](PyoMutableSet[T]):
    """A mutable, ordered collection of unique elements.

    Uses a `dict` as the underlying data structure to maintain insertion order while ensuring uniqueness of elements.

    Thus, it has the same characteristics of "standard" sets, with lookup and iteration speed the same as a `dict`.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the set with.

    Examples:
        ```python
        >>> from pyochain import SetMutOrdered
        >>> s = SetMutOrdered(("a", "b", "c"))
        >>> s
        SetMutOrdered('a', 'b', 'c')
        >>> s.add("d")
        >>> s
        SetMutOrdered('a', 'b', 'c', 'd')
        >>> s.discard("b")
        >>> s
        SetMutOrdered('a', 'c', 'd')

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

    @staticmethod
    def from_ref[V](data: dict[V, Any]) -> SetMutOrdered[V]:  # pyright: ignore[reportExplicitAny]
        """Create a `SetMutOrdered` from a reference to an existing `dict`.

        This method wraps the provided `dict` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `SetMutOrdered` from foreign functions that return `dict` objects.

        Warning:
            Since the `SetMutOrdered` directly references the original `dict`, any modifications made to the `SetMutOrdered` will also affect the original `dict`, and vice versa.

        Args:
            data (dict[V, Any]): The `dict` to wrap.

        Returns:
            SetMutOrdered[V]: A new `SetMutOrdered` instance.

        Example:
            ```python
            >>> from pyochain import SetMutOrdered
            >>> original = {"Alice": 30, "Bob": 25, "Charlie": 35}
            >>> set_obj = SetMutOrdered.from_ref(original)
            >>> set_obj
            SetMutOrdered('Alice', 'Bob', 'Charlie')
            >>> original["David"] = 40
            >>> set_obj
            SetMutOrdered('Alice', 'Bob', 'Charlie', 'David')

            ```
        """
        instance: SetMutOrdered[V] = SetMutOrdered.__new__(SetMutOrdered)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @override
    def add(self, value: T) -> None:
        self._inner[value] = None

    @override
    def discard(self, value: T) -> None:
        del self._inner[value]

    @override
    def intersection(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner.keys() & other)

    @override
    def union(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner.keys() | other)

    @override
    def difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner.keys() - other)

    @override
    def symmetric_difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner.keys() ^ other)


def _set_eq[T](left: SetMut[T] | Set[T], right: object) -> bool:
    """Helper function to compare `Set` and `SetMut` instances for equality.

    Oddly enough, the official doc says that two objects that compare equal must have the same hash.

    But a `set` can compare equal to a `frozenset`, even though the former is not even hashable??

    Args:
        left (SetMut[T] | Set[T]): The left-hand side set to compare.
        right (object): The right-hand side object to compare against.

    Returns:
        bool: `True` if the sets are equal, `False` otherwise.
    """
    match right:
        case Set() | SetMut():
            return left.inner == right.inner  # pyright: ignore[reportUnknownMemberType]
        case frozenset() | set() | SetMutOrdered():
            return left.inner == right
        case _:
            return False
