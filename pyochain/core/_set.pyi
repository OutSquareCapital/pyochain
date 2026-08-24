from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import Self, overload, override

from pyochain.abc import PyoMutableSet, PyoSet

from .protocols import ArgsWrapper

# TODO: address the following note from official python docs regarding Set performance, with benchmarks:
# To override the comparisons (presumably for speed, as the semantics are fixed),
# redefine __le__() and __ge__(), then the other operations will automatically follow suit.

class Set[T](PyoSet[T], ArgsWrapper[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `collections::abc::Collection` Protocol, so it can be used as a standard immutable collection.

    The underlying data structure is a `frozenset`.

    Tip:
        `Set(frozenset)` is a no-copy operation since Python optimizes this under the hood.
    """
    @overload
    def __new__(cls, data: Iterable[T], /) -> Self: ...
    @overload
    def __new__(cls, data: T, /, *more: T) -> Self: ...
    @overload
    def __new__(cls) -> Self: ...
    def __new__(cls, data: Iterable[T] | T = (), /, *more: T) -> Self:
        """Create a new `Set` instance.

        If not arguments are provided, an empty `Set` is created.

        Args:
            data (Iterable[T] | T): Initial elements to populate the set with. Defaults to `()`.
            *more (T): Additional elements to add to the set.

        Returns:
            Self: A new `Set` instance.

        Example:
            ```python
            from pyochain import Set, Iter, Range

            data = (0, 1, 2, 3)

            # Create a `Set` from an iterable
            assert Set(data) == Set(Range(0, 4)) == frozenset(data)

            # Create a `Set` from a single, non-iterable element
            assert Set(1) == Set((1,)) == Set([1]) == frozenset([1])

            # Create a `Set` from multiple elements
            assert Set(0, 1, 2, 3) == Set(data)

            # Create an empty `Set`
            assert Set() == Set([]) == Set(()) == frozenset()
            assert repr(Set()) == "Set()"

            # If you already have a `frozenset`, you can use it directly without copying:
            fs = frozenset(data)
            s2 = Set(fs)
            assert s2 == Set(data)
            find_one: Callable[[object], bool] = lambda x: x == 0
            a = s2.iter().find(find_one).unwrap()
            b = Iter(fs).find(find_one).unwrap()
            assert a is b
            ```
        """

    @override
    def __contains__(self, item: object) -> bool: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @override
    # pyrefly: ignore [bad-override]
    def __and__(self, value: AbstractSet[object], /) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self&value.

        Args:
            value (AbstractSet[object]): The set to perform the intersection with.

        Returns:
            Self: A new instance of the same class containing the intersection of the two sets.

        Example:
            ```python
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(2, 3, 4)
            s3 = s1 & s2
            assert s3 == Set(2, 3)
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def __or__[S](self, value: AbstractSet[S], /) -> Set[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self|value.

        Args:
            value (AbstractSet[S]): The set to perform the union with.

        Returns:
            Set[T | S]: A new `Set` instance containing the union of the two sets.

        Example:
            ```python
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(3, 4, 5)
            s3 = s1 | s2
            assert s3 == Set(1, 2, 3, 4, 5)
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def __sub__(self, value: AbstractSet[object], /) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self-value.

        Args:
            value (AbstractSet[object]): The set to perform the difference with.

        Returns:
            Self: A new instance of the same class containing the difference of the two sets.

        Example:
            ```python
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(2, 3, 4)
            s3 = s1 - s2
            assert s3 == Set(1)
            ```

        """

    @override
    # pyrefly: ignore [bad-override]
    def __xor__[S](self, value: AbstractSet[S], /) -> Set[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self^value.

        Args:
            value (AbstractSet[S]): The set to perform the symmetric difference with.

        Returns:
            Set[T | S]: A new `Set` instance containing the symmetric difference of the two sets.

        Example:
            ```python
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(2, 3, 4)
            s3 = s1 ^ s2
            assert s3 == Set(1, 4)
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
            from pyochain import Set

            s1 = Set(1, 2)
            s2 = Set(1, 2, 3)
            assert s1 <= s2
            assert not s2 <= s1
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
            from pyochain import Set

            s1 = Set(1, 2)
            s2 = Set(1, 2, 3)
            assert s1 < s2
            assert not s2 < s1
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
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(1, 2)
            assert s1 >= s2
            assert not s2 >= s1
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
            from pyochain import Set

            s1 = Set(1, 2, 3)
            s2 = Set(1, 2)
            assert s1 > s2
            assert not s2 > s1
            ```
        """

    @override
    def __eq__(self, value: object, /) -> bool: ...
    @override
    def __hash__(self) -> int: ...
    @override
    @staticmethod
    def wrap[W](iterable: frozenset[W]) -> Set[W]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    @staticmethod
    def from_iter[I](iterable: Iterable[I], /) -> Set[I]: ...
    @override
    @staticmethod
    def of[E](*args: E) -> Set[E]: ...
    @override
    def isdisjoint(self, s: Iterable[object], /) -> bool:
        """Return True if two sets have a null intersection."""

    @override
    def is_subset(self, other: Iterable[object]) -> bool: ...
    @override
    def is_superset(self, other: Iterable[object]) -> bool: ...
    @override
    # pyrefly: ignore [bad-override]
    def intersection(self, *others: Iterable[object]) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    # pyrefly: ignore [bad-override]
    def union[S](self, *others: Iterable[S]) -> Set[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    # pyrefly: ignore [bad-override]
    def difference(self, *others: Iterable[object]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    def symmetric_difference[S](self, other: Iterable[S]) -> Set[T | S]: ...

class SetMut[T](PyoMutableSet[T], ArgsWrapper[T]):
    """A mutable, unordered collection of unique elements.

    Unlike [`Set`][Set] which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `collections::abc::MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    """
    @overload
    def __new__(cls, data: Iterable[T], /) -> Self: ...
    @overload
    def __new__(cls, data: T, *more: T) -> Self: ...
    @overload
    def __new__(cls) -> Self: ...
    def __new__(cls, data: Iterable[T] | T = (), *more: T) -> Self:
        """Create a new `SetMut` instance.

        If no arguments are provided, an empty `SetMut` is created.

        Args:
            data (Iterable[T] | T): Initial elements to populate the set with. Defaults to `()`.
            *more (T): Additional elements to add to the set.

        Returns:
            Self: A new `SetMut` instance.

        Example:
            ```python
            from pyochain import SetMut

            data = (0, 1, 2, 3)

            # Create a `SetMut` from an iterable
            assert SetMut(data) == SetMut(range(0, 4)) == set(data)

            # Create a `SetMut` from a single, non-iterable element
            assert SetMut(1) == SetMut((1,)) == SetMut([1]) == set([1])

            # Create a `SetMut` from multiple elements
            assert SetMut(0, 1, 2, 3) == SetMut(data)

            # Create an empty `SetMut`
            assert SetMut() == SetMut([]) == SetMut(()) == set()
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
    @override
    # pyrefly: ignore [bad-override]
    def __and__(self, value: AbstractSet[object], /) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self&value."""

    @override
    def __iand__(self, value: AbstractSet[object], /) -> SetMut[T]:
        """Return self&=value."""

    @override
    # pyrefly: ignore [bad-override]
    def __or__[S](self, value: AbstractSet[S], /) -> SetMut[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self|value.

        Args:
            value (AbstractSet[S]): The set to perform the union with.

        Returns:
            SetMut[T | S]: A new `SetMut` instance containing the union of the two sets.

        Example:
            ```python
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(3, 4, 5)
            s3 = s1 | s2
            assert s3 == SetMut(1, 2, 3, 4, 5)
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
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(3, 4, 5)
            s1 |= s2
            assert s1 == SetMut(1, 2, 3, 4, 5)
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def __sub__(self, value: AbstractSet[object], /) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self-value.

        Args:
            value (AbstractSet[object]): The set to subtract.

        Returns:
            SetMut[T]: A new `SetMut` instance containing the result of the subtraction.

        Example:
            ```python
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(2, 3, 4)
            s3 = s1 - s2
            assert s3 == SetMut(1)
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
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(2, 3, 4)
            s1 -= s2
            assert s1 == SetMut(1)
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def __xor__[S](self, value: AbstractSet[S], /) -> SetMut[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return self^value.

        Args:
            value (AbstractSet[S]): The set to perform the symmetric difference with.

        Returns:
            SetMut[T | S]: A new `SetMut` instance containing the result of the symmetric difference.

        Example:
            ```python
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(3, 4, 5)
            s3 = s1 ^ s2
            assert s3 == SetMut(1, 2, 4, 5)
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
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(3, 4, 5)
            s1 ^= s2
            assert s1 == SetMut(1, 2, 4, 5)
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
            from pyochain import SetMut

            s1 = SetMut(1, 2)
            s2 = SetMut(1, 2, 3)
            assert s1 <= s2
            assert not s2 <= s1
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
            from pyochain import SetMut

            s1 = SetMut(1, 2)
            s2 = SetMut(1, 2, 3)
            assert s1 < s2
            assert not s2 < s1
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
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(1, 2)
            assert s1 >= s2
            assert not s2 >= s1
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
            from pyochain import SetMut

            s1 = SetMut(1, 2, 3)
            s2 = SetMut(1, 2)
            assert s1 > s2
            assert not s2 > s1
            ```
        """

    @override
    @staticmethod
    def wrap[W](iterable: set[W]) -> SetMut[W]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    @staticmethod
    def from_iter[I](iterable: Iterable[I], /) -> SetMut[I]: ...
    @override
    @staticmethod
    def of[E](*args: E) -> SetMut[E]: ...
    @override
    def add(self, value: T) -> None:
        """Add an element to **self**.

        Args:
            value (T): The element to add.

        Example:
            ```python
            from pyochain import SetMut, Vec

            s = SetMut("a", "b")
            s.add("c")
            assert s.iter().sort() == Vec("a", "b", "c")
            ```
        """

    def copy(self) -> Self:
        """Create a shallow copy of the underlying `set`.

        Returns:
            Self: A shallow copy of the underlying `set`.

        Example:
            ```python
            from pyochain import SetMut

            s = SetMut("a", "b")
            s_copy = s.copy()
            s_copy.add("c")
            assert not "c" in s
            assert "c" in s_copy
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
            from pyochain import SetMut, Vec

            s = SetMut("a", "b", "c")
            s.discard("b")
            assert s.iter().sort() == Vec("a", "c")
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
    # pyrefly: ignore [bad-override]
    def intersection(self, *others: Iterable[object]) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @override
    # pyrefly: ignore [bad-override]
    def union[S](self, *others: Iterable[S]) -> SetMut[T | S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def update(self, *s: Iterable[T]) -> None:
        """Update the set, adding elements from all others."""

    @override
    # pyrefly: ignore [bad-override]
    def difference(self, *others: Iterable[object]) -> SetMut[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def difference_update(self, *s: Iterable[object]) -> None:
        """Update the set, removing elements found in others."""

    @override
    def symmetric_difference[S](self, other: Iterable[S]) -> SetMut[T | S]: ...
