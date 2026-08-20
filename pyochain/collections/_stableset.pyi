from collections.abc import Iterable, Iterator
from typing import Any, Self, override

from pyochain import SetMut
from pyochain.abc import PyoMutableSet

class StableSet[T](PyoMutableSet[T]):
    """A mutable collection of unique elements which remember their insertion order.

    Uses a `dict` as the underlying data structure to maintain insertion order while ensuring uniqueness of elements.

    Thus, it has the same characteristics of "standard" sets, with lookup and iteration speed the same as a `dict`.

    This is very similar to using `Dict::from_keys` with `None` values, but with a specialized interface for set operations.

    Note:
        This is not the same as `sortedcontainers`, i.e it does not maintain the elements in sorted order, but rather in the order they were inserted.

    Example:
        ```python
        from pyochain import Some
        from pyochain.collections import StableSet

        s = StableSet("a", "b", "c")
        assert repr(s) == "StableSet('a', 'b', 'c')"

        # Mutation preserves insertion order

        s.add("d")
        assert s.iter().next() == Some("a")
        assert s.iter().last() == "d"
        assert repr(s) == "StableSet('a', 'b', 'c', 'd')"

        s.discard("b")
        assert s.iter().next() == Some("a")
        assert s.iter().skip(1).next() == Some("c")
        assert s.iter().last() == "d"
        assert repr(s) == "StableSet('a', 'c', 'd')"
        ```
    """
    def __new__(cls, data: Iterable[T] | T = (), /, *elements: T) -> Self:
        """Create a new `StableSet` instance.

        If data is
        - not provided, or an empty `Iterable`, an empty `StableSet` is created.
        - a non-empty `Iterable`, the elements of the iterable are added to the set.
        - a single non-iterable element, it creates a `StableSet` with that element.

        Additional elements can be provided as positional arguments.

        Args:
            data (Iterable[T] | T): initial data to populate the `StableSet`. Defaults to `()`.
            *elements (T): Additional elements to add to the set.

        Examples:
            ```python
            from pyochain.collections import StableSet

            data = ("a", "b", "c")
            # Creates an empty `StableSet`
            assert StableSet() == StableSet(()) == StableSet([]) == frozenset()
            # Create a `StableSet` from an iterable
            assert StableSet(data) == StableSet(list(data)) == frozenset(data)
            # Create a `StableSet` from a single non-iterable element
            assert StableSet(1) == StableSet([1]) == frozenset([1])
            # Create a `StableSet` from multiple elements
            assert StableSet("a", "b", "c") == StableSet(*data) == frozenset(data)
            ```
        """
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __contains__(self, item: object) -> bool: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @staticmethod
    def from_ref[V](data: dict[V, Any]) -> StableSet[V]:
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
            from pyochain.collections import StableSet

            original = {"Alice": 30, "Bob": 25, "Charlie": 35}
            set_obj = StableSet.from_ref(original)
            assert set_obj == StableSet("Alice", "Bob", "Charlie")
            original["David"] = 40
            assert set_obj == StableSet("Alice", "Bob", "Charlie", "David")
        ```
        """
    @override
    def add(self, value: T) -> None: ...
    def copy(self) -> StableSet[T]:
        """Return a shallow copy of the `StableSet`.

        Returns:
            StableSet[T]: A new `StableSet` instance that is a shallow copy of the original.

        Example:
            ```python
            from pyochain.collections import StableSet

            s = StableSet("a", "b", "c")
            s_copy = s.copy()
            assert s_copy == StableSet("a", "b", "c")
            ```
        """

    @override
    def discard(self, value: T) -> None: ...
    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[T]: ...
    @override
    def union[S](self, other: Iterable[S]) -> SetMut[T | S]: ...
    @override
    def difference(self, other: Iterable[Any]) -> SetMut[T]: ...
    @override
    def symmetric_difference[S](self, other: Iterable[S]) -> SetMut[T | S]: ...
