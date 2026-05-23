from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, MutableSet
from collections.abc import Set as AbstractSet
from typing import Final, Self, override

from ._utils import get_repr
from .abc import PyoCollection, PyoSet


class BaseConcreteSet[T](ABC):
    """Internal mixin for concrete set-like classes.

    It provides a common interface for documentation, DRY code, and type consistency, as well as enforcing a *you must implement concrete methods for operator dunders* policy for set operations.

    pyochain philosophy is to prefer explicit method calls that read like natural language instead of obscure operators.

    Concrete set classes should inherit from this base class and implement it's abstract methods.
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @abstractmethod
    def intersection(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing only elements present in both sets.

        If the sets have no common elements, the result is empty.

        This operation is commutative: `A.intersection(B) == B.intersection(A)`.


        Args:
            other (AbstractSet[T]): The set to intersect with.

        Returns:
            AbstractSet[T]: A new `Set` containing shared elements only.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> from_set = Set((1, 2))
            >>> from_set.intersection({2, 3})
            Set(2,)
            >>> from_set.intersection({3, 4})
            Set()
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().intersection({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('b', 'c')
            >>> from_items = (
            ...     dct.items()
            ...     .intersection({("b", 2), ("c", 3), ("d", 4)})
            ...     .iter()
            ...     .sort()
            ... )
            >>> from_items
            Vec(('b', 2), ('c', 3))

            ```
        """

    @abstractmethod
    def union(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing all unique elements from both sets.

        This operation is commutative: `A.union(B) == B.union(A)`.

        Args:
            other (AbstractSet[T]): The set to combine with.

        Returns:
            AbstractSet[T]: A new set containing all elements from **self** and **other**.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).union({2, 3}).union({4}).iter().sort()
            Vec(1, 2, 3, 4)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().union({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('a', 'b', 'c', 'd')
            >>> from_items = (
            ...     dct.items().union({("b", 2), ("c", 3), ("d", 4)}).iter().sort()
            ... )
            >>> from_items
            Vec(('a', 1), ('b', 2), ('c', 3), ('d', 4))

            ```
        """

    @abstractmethod
    def difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in this set but not in `other`.

        The result contains every element that is in this set EXCEPT those that are also present in `other`.

        This operation is NOT commutative.

        Args:
            other (AbstractSet[T]): The set whose elements should be excluded.

        Returns:
            AbstractSet[T]: A new set containing elements unique to this set.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).difference({2, 3})
            Set(1,)
            >>> Set((1, 2)).difference({3, 4}).iter().sort()
            Vec(1, 2)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().difference({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('a')
            >>> from_items = (
            ...     dct.items().difference({("b", 2), ("c", 3), ("d", 4)}).iter().sort()
            ... )
            >>> from_items
            Vec(('a', 1))

            ```
        """

    @abstractmethod
    def symmetric_difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in either set but not in both.

        The result contains elements that are in this set XOR `other`—i.e., elements present in one set but not in both.

        This is the opposite of [`Set::intersection`][Set.intersection].

        This operation is commutative: `A.symmetric_difference(B) == B.symmetric_difference(A)`.

        Args:
            other (AbstractSet[T]): The set to compute symmetric difference with.

        Returns:
            AbstractSet[T]: A new set containing elements unique to each set.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).symmetric_difference({2, 3}).iter().sort()
            Vec(1, 3)
            >>> Set((1, 2, 3)).symmetric_difference({3, 4, 5}).iter().sort()
            Vec(1, 2, 4, 5)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = (
            ...     dct.keys().symmetric_difference({"b", "c", "d"}).iter().sort()
            ... )
            >>> from_keys
            Vec('a', 'd')
            >>> from_items = (
            ...     dct.items()
            ...     .symmetric_difference({("b", 2), ("c", 3), ("d", 4)})
            ...     .iter()
            ...     .sort()
            ... )
            >>> from_items
            Vec(('a', 1), ('d', 4))

            ```
        """


class Set[T](PyoSet[T], BaseConcreteSet[T]):
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
    __match_args__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]
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


class SetMut[T](BaseConcreteSet[T], MutableSet[T], PyoCollection[T]):  # noqa: PLW1641
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
        instance: SetMut[V] = object.__new__(SetMut)  # pyright: ignore[reportUnknownVariableType]
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
        case frozenset() | set():
            return left.inner == right
        case _:
            return False
