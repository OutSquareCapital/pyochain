from collections.abc import Callable, Iterable, Iterator
from typing import Self, SupportsIndex, final, overload, override

from _typeshed import SupportsRichComparison

from pyochain.abc import PyoMutableSequence

from .protocols import ArgsWrapper

type IntoVec[T] = Vec[T] | list[T]

@final
class Vec[T](PyoMutableSequence[T], ArgsWrapper[T]):
    """Represent a mutable sequence of elements.

    Implement `collections::abc::MutableSequence`, and pyochain's `PyoMutableSequence` ABC.

    Unlike [`Seq`][Seq] which is immutable, `Vec` allows in-place modification of elements.

    As such, `Vec` is more suitable when you need to build up a collection incrementally, or when you need to perform many modifications on the collection.

    On the other hand, [`Seq`][Seq] is more memory efficient when you have a fixed collection that doesn't require modification.

    This is due to the fact that CPython don't have to allocate extra space to account for potential future modifications.

    It uses a `list` as the underlying data structure, so it has the same performance characteristics regarding indexing, slicing, and iteration.

    """
    @overload
    def __new__(cls, data: Iterable[T], /) -> Self: ...
    @overload
    def __new__(cls, data: T, /, *more: T) -> Self: ...
    @overload
    def __new__(cls, /) -> Self: ...
    def __new__(cls, data: Iterable[T] | T = (), /, *more: T) -> Self:
        """Create a new `Vec` instance.

        If not arguments are provided, an empty `Vec` is created.

        Args:
            data (Iterable[T] | T): The data to initialize the `Vec` with. Defaults to `()`.
            *more (T): Additional elements to include in the `Vec`.

        Returns:
            Self: A new `Vec` instance.

        Example:
            ```python
            from pyochain import Vec

            py_list = [1, 2, 3]

            # Create a Vec from an iterable
            assert Vec(iter(py_list)) == Vec(py_list)
            # Create a Vec from individual elements
            assert Vec(1, 2, 3) == py_list
            # Create an empty Vec
            assert Vec() == Vec([]) == Vec(()) == []
            # Creating a Vec from a list will copy the underlying data
            vec = Vec(py_list)
            vec[0] = 10
            assert py_list == [1, 2, 3]
            ```
        """

    @override
    def __iter__(self) -> Iterator[T]: ...
    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, s: slice[SupportsIndex | None], /) -> Vec[T]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | Vec[T]: ...
    @overload
    def __setitem__(self, key: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(
        self, key: slice[SupportsIndex | None], value: Iterable[T]
    ) -> None: ...
    @override
    def __setitem__(
        self, key: SupportsIndex | slice[SupportsIndex | None], value: T | Iterable[T]
    ) -> None: ...
    @override
    def __delitem__(self, key: SupportsIndex | slice[SupportsIndex | None]) -> None: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __add__[V](self: Vec[V], value: IntoVec[V], /) -> Vec[V]: ...
    @overload
    def __add__[S](self, value: IntoVec[S], /) -> Vec[S | T]: ...
    def __add__[V, S](
        self: Vec[V], value: IntoVec[V] | IntoVec[S], /
    ) -> Vec[V] | Vec[S | V]: ...
    @override
    def __iadd__(self, value: Iterable[T], /) -> Vec[T]: ...
    def __mul__(self, value: SupportsIndex, /) -> Vec[T]: ...
    def __rmul__(self, value: SupportsIndex, /) -> Vec[T]: ...
    def __imul__(self, value: SupportsIndex, /) -> Vec[T]: ...
    @override
    def __contains__(self, key: object, /) -> bool: ...
    def __gt__(self, value: IntoVec[T], /) -> bool: ...
    def __ge__(self, value: IntoVec[T], /) -> bool: ...
    def __lt__(self, value: IntoVec[T], /) -> bool: ...
    def __le__(self, value: IntoVec[T], /) -> bool: ...
    @override
    def __reversed__(self) -> Iterator[T]: ...
    @override
    @staticmethod
    def of[E](*elements: E) -> Vec[E]: ...
    @override
    @staticmethod
    def from_iter[I](iterable: Iterable[I], /) -> Vec[I]: ...
    @staticmethod
    @override
    def wrap[S](iterable: list[S]) -> Vec[S]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def reverse(self) -> None: ...
    @override
    def append(self, value: T) -> None: ...
    @override
    def extend(self, iterable: Iterable[T]) -> None: ...
    @override
    def clear(self) -> None: ...
    def copy(self) -> Self:
        """Return a shallow copy of the `Vec`.

        This is equivalent to `list_1.copy()` for standard lists.

        Returns:
            Self: A new `Vec` instance with the same elements.

        Example:
            ```python
            from pyochain import Vec

            v1 = Vec(1, 2, 3)
            v2 = v1.copy()
            assert v2 == Vec(1, 2, 3)
            assert v1 is not v2
            ```
        """

    def repeat(self, n: int) -> Vec[T]:
        """Repeat the elements of the `Vec` **n** times and return a new `Vec`.

        This is equivalent to `list_1 * n` for standard lists.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Vec[T]: The new `Vec` after repetition.

        See Also:
            [`Vec::repeat_mut`][repeat_mut] which modifies the `Vec` in place.

        Example:
            ```python
            from pyochain import Vec

            v = Vec(1, 2, 3).repeat(2)
            assert v == Vec(1, 2, 3, 1, 2, 3)
            ```
        """

    def repeat_mut(self, n: int) -> Self:
        """Repeat the elements of the `Vec` in place.

        This is equivalent to `list_1 *= n` for standard lists.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Self: The modified `Vec` after repetition (self).

        See Also:
            [`Vec::repeat`][repeat] which returns a new `Vec` (copy).

        Example:
            ```python
            from pyochain import Vec

            vec = Vec(1, 2, 3).repeat_mut(2)
            assert vec == Vec(1, 2, 3, 1, 2, 3)
            ```
        """

    @override
    def insert(self, index: int, value: T) -> None:
        """Inserts an element at position index within the vector, shifting all elements after it to the right.

        Args:
            index (int): Position where to insert the element.
            value (T): The element to insert.

        Example:
            ```python
            from pyochain import Vec

            vec = Vec("a", "b", "c")
            vec.insert(1, "d")
            assert vec == Vec("a", "d", "b", "c")
            vec.insert(4, "e")
            assert vec == Vec("a", "d", "b", "c", "e")
            ```
        """

    def sort[U: SupportsRichComparison](
        self: Vec[U], *, reverse: bool = False
    ) -> Vec[U]:
        """Sort the elements of the `Vec` in place.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            reverse (bool): If `True`, sort in descending order.

        Returns:
            Vec[U]: The sorted `Vec` instance (self).

        Example:
            ```python
            from pyochain import Vec, Iter

            x = Vec(3, 1, 2).sort()
            assert x == Vec(1, 2, 3)
            ```
        """

    def sort_by(
        self, key: Callable[[T], SupportsRichComparison], *, reverse: bool = False
    ) -> Self:
        """Sort the elements of the `Vec`  in place with a key function.

        The `key` function is applied to each element before sorting, and the results are used for comparison.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            key (Callable[[T], SupportsRichComparison]): function to extract a comparison key from each element.
            reverse (bool): If True, sort in descending order.

        Returns:
            Self: The sorted `Vec` instance (self).

        Example:
            ```python
            from pyochain import Vec, Iter

            x = Vec("3", "1", "2").sort_by(int)
            y = Vec("1", "2", "3")
            assert x == y
            ```
        """

    def concat(self, other: IntoVec[T]) -> Vec[T]:
        """Concatenate another `Vec` or `list` to **self** and return a new `Vec`.

        Note:
            This is equivalent to `list_1 + list_2` for standard lists.

        Args:
            other (IntoVec[T]): The other `Vec` to concatenate.

        Returns:
            Vec[T]: The new `Vec` after concatenation.

        See Also:
            [`Vec::concat_mut`][concat_mut] which modifies **self** in place.

        Example:
            ```python
            from pyochain import Vec

            v1 = Vec(1, 2, 3)
            v2 = [4, 5, 6]  # Can also concatenate a standard list
            expected = Vec(1, 2, 3, 4, 5, 6)
            v3 = v1.concat(v2)
            assert v3 == expected
            v1.clear()  # Clean up the original vec
            assert v1 == Vec()
            # New vec remains unaffected
            assert v3 == expected
            ```
        """

    def concat_mut(self, other: IntoVec[T]) -> Self:
        """Concatenate another `Vec` or `list` to **self** in place.

        This is equivalent to `list_1 += list_2` for standard lists.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            other (IntoVec[T]): The other `Vec` to concatenate.

        Returns:
            Self: The modified `Vec` after concatenation (self).

        See Also:
            - [`Vec::concat`][concat] which returns a new `Vec` (copy).
            - [`Vec::extend`][extend] which can take any `Iterable`.

        Example:
            ```python
            from pyochain import Vec

            v1 = Vec(1, 2, 3)
            v2 = [4, 5, 6]  # Can also concatenate a standard list
            expected = Vec(1, 2, 3, 4, 5, 6)
            assert v1.concat_mut(v2) == expected
            assert v1 == expected
            ```
        """
