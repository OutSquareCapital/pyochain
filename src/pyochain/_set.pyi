from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import Final, Self, override

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

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]
    inner: Final[frozenset[T]]

    def __init__(self, data: Iterable[T]) -> None: ...
    @override
    def __contains__(self, item: object) -> bool: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __and__(self, value: AbstractSet[object], /) -> Self:
        """Return self&value.

        Args:
            value (AbstractSet[object]): The set to perform the intersection with.

        Returns:
            Self: A new instance of the same class containing the intersection of the two sets.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((2, 3, 4))
            >>> s3 = s1 & s2
            >>> s3
            Set(2, 3)

            ```
        """

    @override
    def __or__[S](self, value: AbstractSet[S], /) -> Set[T | S]:
        """Return self|value.

        Args:
            value (AbstractSet[S]): The set to perform the union with.

        Returns:
            Set[T | S]: A new `Set` instance containing the union of the two sets.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((3, 4, 5))
            >>> s3 = s1 | s2
            >>> s3
            Set(1, 2, 3, 4, 5)

            ```
        """

    @override
    def __sub__(self, value: AbstractSet[object], /) -> Self:
        """Return self-value.

        Args:
            value (AbstractSet[object]): The set to perform the difference with.

        Returns:
            Self: A new instance of the same class containing the difference of the two sets.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((2, 3, 4))
            >>> s3 = s1 - s2
            >>> s3
            Set(1,)

            ```

        """

    @override
    def __xor__[S](self, value: AbstractSet[S], /) -> Set[T | S]:
        """Return self^value.

        Args:
            value (AbstractSet[S]): The set to perform the symmetric difference with.

        Returns:
            Set[T | S]: A new `Set` instance containing the symmetric difference of the two sets.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((2, 3, 4))
            >>> s3 = s1 ^ s2
            >>> s3
            Set(1, 4)

            ```
        """

    @override
    def __le__(self, value: AbstractSet[object], /) -> bool:
        """Return self<=value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a subset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2))
            >>> s2 = Set((1, 2, 3))
            >>> s1 <= s2
            True
            >>> s2 <= s1
            False

            ```
        """
    @override
    def __lt__(self, value: AbstractSet[object], /) -> bool:
        """Return self<value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a proper subset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2))
            >>> s2 = Set((1, 2, 3))
            >>> s1 < s2
            True
            >>> s2 < s1
            False

            ```
        """

    @override
    def __ge__(self, value: AbstractSet[object], /) -> bool:
        """Return self>=value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a superset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((1, 2))
            >>> s1 >= s2
            True
            >>> s2 >= s1
            False

            ```
        """

    @override
    def __gt__(self, value: AbstractSet[object], /) -> bool:
        """Return self>value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a proper superset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> s1 = Set((1, 2, 3))
            >>> s2 = Set((1, 2))
            >>> s1 > s2
            True
            >>> s2 > s1
            False

            ```
        """

    @override
    def __eq__(self, value: object, /) -> bool: ...
    @override
    def __hash__(self) -> int: ...
    @override
    def isdisjoint(self, s: Iterable[object], /) -> bool:
        """Return True if two sets have a null intersection."""
    @override
    def is_subset(self, other: Iterable[object]) -> bool: ...
    @override
    def is_superset(self, other: Iterable[object]) -> bool: ...
    @override
    def intersection(self, *others: Iterable[object]) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def union[S](self, *others: Iterable[S]) -> Set[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    def difference(self, *others: Iterable[object]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    def symmetric_difference[S](self, other: Iterable[S]) -> Set[T | S]: ...

class SetMut[T](PyoMutableSet[T]):
    """A mutable, unordered collection of unique elements.

    Unlike [`Set`][Set] which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `collections::abc::MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    Tip:
        If you have an existing `set`, consider using [`SetMut::from_ref`][from_ref] to avoid unnecessary copying.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the set with.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]
    inner: Final[set[T]]

    def __init__(self, data: Iterable[T]) -> None: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __contains__(self, item: object) -> bool: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @override
    def __and__(self, value: AbstractSet[object], /) -> SetMut[T]:
        """Return self&value."""

    @override
    def __iand__(self, value: AbstractSet[object], /) -> SetMut[T]:
        """Return self&=value."""
    @override
    def __or__[S](self, value: AbstractSet[S], /) -> SetMut[T | S]:
        """Return self|value.

        Args:
            value (AbstractSet[S]): The set to perform the union with.

        Returns:
            SetMut[T | S]: A new `SetMut` instance containing the union of the two sets.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((3, 4, 5))
            >>> s3 = s1 | s2
            >>> s3
            SetMut(1, 2, 3, 4, 5)

            ```
        """
    @override
    def __ior__(self, value: AbstractSet[T], /) -> SetMut[T]:
        """Return self|=value.

        Args:
            value (AbstractSet[T]): The set to perform the union with.

        Returns:
            SetMut[T]: The current instance after performing the union operation.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((3, 4, 5))
            >>> s1 |= s2
            >>> s1
            SetMut(1, 2, 3, 4, 5)

            ```
        """

    @override
    def __sub__(self, value: AbstractSet[object], /) -> SetMut[T]:
        """Return self-value.

        Args:
            value (AbstractSet[object]): The set to subtract.

        Returns:
            SetMut[T]: A new `SetMut` instance containing the result of the subtraction.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((2, 3, 4))
            >>> s3 = s1 - s2
            >>> s3
            SetMut(1,)

            ```
        """

    @override
    def __isub__(self, value: AbstractSet[object], /) -> SetMut[T]:
        """Return self-=value.

        Args:
            value (AbstractSet[object]): The set to subtract.

        Returns:
            SetMut[T]: The current instance after performing the subtraction operation.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((2, 3, 4))
            >>> s1 -= s2
            >>> s1
            SetMut(1,)

            ```
        """

    @override
    def __xor__[S](self, value: AbstractSet[S], /) -> SetMut[T | S]:
        """Return self^value.

        Args:
            value (AbstractSet[S]): The set to perform the symmetric difference with.

        Returns:
            SetMut[T | S]: A new `SetMut` instance containing the result of the symmetric difference.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((3, 4, 5))
            >>> s3 = s1 ^ s2
            >>> s3
            SetMut(1, 2, 4, 5)

            ```
        """

    @override
    def __ixor__(self, value: AbstractSet[T], /) -> SetMut[T]:
        """Return self^=value.

        Args:
            value (AbstractSet[T]): The set to perform the symmetric difference with.

        Returns:
            SetMut[T]: The current instance after performing the symmetric difference operation.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((3, 4, 5))
            >>> s1 ^= s2
            >>> s1
            SetMut(1, 2, 4, 5)

            ```
        """

    @override
    def __le__(self, value: AbstractSet[object], /) -> bool:
        """Return self<=value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a subset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2))
            >>> s2 = SetMut((1, 2, 3))
            >>> s1 <= s2
            True
            >>> s2 <= s1
            False

            ```
        """

    @override
    def __lt__(self, value: AbstractSet[object], /) -> bool:
        """Return self<value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a proper subset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2))
            >>> s2 = SetMut((1, 2, 3))
            >>> s1 < s2
            True
            >>> s2 < s1
            False

            ```
        """

    @override
    def __ge__(self, value: AbstractSet[object], /) -> bool:
        """Return self>=value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a superset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((1, 2))
            >>> s1 >= s2
            True
            >>> s2 >= s1
            False

            ```
        """

    @override
    def __gt__(self, value: AbstractSet[object], /) -> bool:
        """Return self>value.

        Args:
            value (AbstractSet[object]): The set to compare against.

        Returns:
            bool: `True` if self is a proper superset of value, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s1 = SetMut((1, 2, 3))
            >>> s2 = SetMut((1, 2))
            >>> s1 > s2
            True
            >>> s2 > s1
            False

            ```
        """

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

    def copy(self) -> Self:
        """Create a shallow copy of the underlying `set`.

        Returns:
            Self: A shallow copy of the underlying `set`.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s = SetMut(("a", "b"))
            >>> s_copy = s.copy()
            >>> s_copy.add("c")
            >>> "c" in s
            False
            >>> "c" in s_copy
            True

            ```
        """

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

    def intersection_update(self, *s: Iterable[object]) -> None:
        """Update the set, keeping only elements found in it and all others."""

    @override
    def isdisjoint(self, s: Iterable[object], /) -> bool:
        """Return True if two sets have a null intersection."""

    @override
    def is_subset(self, other: Iterable[object]) -> bool: ...
    @override
    def is_superset(self, other: Iterable[object]) -> bool: ...
    @override
    def remove(self, element: T, /) -> None:
        """Remove an element from a set; it must be a member.

        If the element is not a member, raise a KeyError.
        """

    def symmetric_difference_update(self, s: Iterable[T], /) -> None:
        """Update the set, keeping only elements found in either set, but not in both."""

    @override
    def intersection(self, *others: Iterable[object]) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    def union[S](self, *others: Iterable[S]) -> SetMut[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def update(self, *s: Iterable[T]) -> None:
        """Update the set, adding elements from all others."""

    @override
    def difference(self, *others: Iterable[object]) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def difference_update(self, *s: Iterable[object]) -> None:
        """Update the set, removing elements found in others."""
    @override
    def symmetric_difference[S](self, other: Iterable[S]) -> SetMut[T | S]: ...
