from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, SupportsIndex, overload, override

from ._types import SupportsRichComparison
from ._utils import get_repr, no_doctest
from .abc import PyoMutableSequence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


class Vec[T](PyoMutableSequence[T]):  # noqa: PLW1641
    """Represent a mutable sequence of elements.

    Implement `collections::abc::MutableSequence`, and pyochain's `PyoMutableSequence` ABC.

    Unlike [`Seq`][Seq] which is immutable, `Vec` allows in-place modification of elements.

    As such, `Vec` is more suitable when you need to build up a collection incrementally, or when you need to perform many modifications on the collection.

    On the other hand, [`Seq`][Seq] is more memory efficient when you have a fixed collection that doesn't require modification.

    This is due to the fact that CPython don't have to allocate extra space to account for potential future modifications.

    It uses a `list` as the underlying data structure, so it has the same performance characteristics regarding indexing, slicing, and iteration.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the `Vec` with. If the input is already a `list`, it will be used directly without copying.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: list[T]

    def __init__(self, data: Iterable[T]) -> None:
        match data:
            case list():
                self._inner = data
            case _:
                self._inner = list(data)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner)})"

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, s: slice[SupportsIndex | None], /) -> list[T]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | list[T]:
        return self._inner[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        # pyrefly: ignore[no-matching-overload]
        return self._inner.__setitem__(index, value)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType]

    @override
    def __delitem__(self, index: int | slice) -> None:
        del self._inner[index]

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def __eq__(self, other: object) -> bool:
        match other:
            case Vec():
                return self._inner == other._inner  # pyright: ignore[reportUnknownMemberType]
            case list():
                return self._inner == other
            case _:
                return False

    @property
    @no_doctest
    def inner(self) -> list[T]:
        """The underlying `list` data structure.

        Useful when interoperating with functions that require a standard Python `list`.

        Returns:
            list[T]: The underlying list.
        """
        return self._inner

    def copy(self) -> Self:
        """Return a shallow copy of the `Vec`.

        This is equivalent to `list_1.copy()` for standard lists.

        Returns:
            Self: A new `Vec` instance with the same elements.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v1 = Vec([1, 2, 3])
            >>> v2 = v1.copy()
            >>> v2
            Vec(1, 2, 3)
            >>> v1 is v2
            False

            ```
        """
        return self.__class__(self._inner.copy())

    def repeat(self, n: int) -> Self:
        """Repeat the elements of the `Vec` **n** times and return a new `Vec`.

        This is equivalent to `list_1 * n` for standard lists.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Self: The new `Vec` after repetition.

        See Also:
            [`Vec::repeat_mut`][repeat_mut] which modifies the `Vec` in place.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> Vec([1, 2, 3]).repeat(2)
            Vec(1, 2, 3, 1, 2, 3)

            ```
        """
        return self.__class__(self._inner * n)

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
            >>> from pyochain import Vec
            >>> vec = Vec([1, 2, 3])
            >>> vec.repeat_mut(2)
            Vec(1, 2, 3, 1, 2, 3)
            >>> vec
            Vec(1, 2, 3, 1, 2, 3)

            ```
        """
        self._inner *= n
        return self

    @override
    def insert(self, index: int, value: T) -> None:
        """Inserts an element at position index within the vector, shifting all elements after it to the right.

        Args:
            index (int): Position where to insert the element.
            value (T): The element to insert.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> vec = Vec(["a", "b", "c"])
            >>> vec.insert(1, "d")
            >>> vec
            Vec('a', 'd', 'b', 'c')
            >>> vec.insert(4, "e")
            >>> vec
            Vec('a', 'd', 'b', 'c', 'e')

            ```
        """
        self._inner.insert(index, value)

    def sort[U: SupportsRichComparison[Any]](
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
            >>> from pyochain import Vec, Iter
            >>> Vec([3, 1, 2]).sort()
            Vec(1, 2, 3)

            ```
        """
        self._inner.sort(reverse=reverse)
        return self

    def sort_by(
        self, key: Callable[[T], SupportsRichComparison[Any]], *, reverse: bool = False
    ) -> Self:
        """Sort the elements of the `Vec`  in place with a key function.

        The `key` function is applied to each element before sorting, and the results are used for comparison.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            key (Callable[[T], SupportsRichComparison[Any]]): function to extract a comparison key from each element.
            reverse (bool): If True, sort in descending order.

        Returns:
            Self: The sorted `Vec` instance (self).

        Example:
            ```python
            >>> from pyochain import Vec, Iter
            >>> Vec(["3", "1", "2"]).sort_by(int)
            Vec('1', '2', '3')

            ```
        """
        self._inner.sort(key=key, reverse=reverse)
        return self

    def concat(self, other: list[T] | Self) -> Self:
        """Concatenate another `Vec` or `list` to **self** and return a new `Vec`.

        Note:
            This is equivalent to `list_1 + list_2` for standard lists.

        Args:
            other (list[T] | Self): The other `Vec` to concatenate.

        Returns:
            Self: The new `Vec` after concatenation.

        See Also:
            [`Vec::concat_mut`][concat_mut] which modifies **self** in place.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v1 = Vec([1, 2, 3])
            >>> v2 = [4, 5, 6]  # Can also concatenate a standard list
            >>> v3 = v1.concat(v2)
            >>> v3
            Vec(1, 2, 3, 4, 5, 6)
            >>> v1.clear()  # Clean up the original vec
            >>> v1
            Vec()
            >>> # New vec remains unaffected
            >>> v3
            Vec(1, 2, 3, 4, 5, 6)

            ```
        """
        match other:
            case Vec():
                data = self._inner + other._inner
            case list():
                data = self._inner + other
        return self.__class__(data)

    def concat_mut(self, other: list[T] | Self) -> Self:
        """Concatenate another `Vec` or `list` to **self** in place.

        This is equivalent to `list_1 += list_2` for standard lists.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            other (list[T] | Self): The other `Vec` to concatenate.

        Returns:
            Self: The modified `Vec` after concatenation (self).

        See Also:
            - [`Vec::concat`][concat] which returns a new `Vec` (copy).
            - `Vec::extend` which can take any `Iterable`.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v1 = Vec([1, 2, 3])
            >>> v2 = [4, 5, 6]  # Can also concatenate a standard list
            >>> v1.concat_mut(v2)
            Vec(1, 2, 3, 4, 5, 6)
            >>> v1
            Vec(1, 2, 3, 4, 5, 6)

            ```
        """
        match other:
            case Vec():
                self._inner += other._inner
            case list():
                self._inner += other
        return self

    @override
    def __reversed__(self) -> Iterator[T]:
        return reversed(self._inner)
