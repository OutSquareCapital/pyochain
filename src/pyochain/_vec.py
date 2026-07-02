from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, SupportsIndex, overload, override

from ._types import SupportsRichComparison
from ._utils import get_repr, no_doctest
from .abc import PyoMutableSequence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

type IntoVec[T] = Vec[T] | list[T]


class Vec[T](PyoMutableSequence[T]):  # noqa: PLW1641
    """Represent a mutable sequence of elements.

    Implement `collections::abc::MutableSequence`, and pyochain's `PyoMutableSequence` ABC.

    Unlike [`Seq`][Seq] which is immutable, `Vec` allows in-place modification of elements.

    As such, `Vec` is more suitable when you need to build up a collection incrementally, or when you need to perform many modifications on the collection.

    On the other hand, [`Seq`][Seq] is more memory efficient when you have a fixed collection that doesn't require modification.

    This is due to the fact that CPython don't have to allocate extra space to account for potential future modifications.

    It uses a `list` as the underlying data structure, so it has the same performance characteristics regarding indexing, slicing, and iteration.

    Args:
        data (Iterable[T]): Any `Iterable` of elements to initialize the `Vec` with.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: list[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = list(data)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

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

    @overload
    def __add__[V](self: Vec[V], value: IntoVec[V], /) -> Vec[V]: ...

    @overload
    def __add__[S](self, value: IntoVec[S], /) -> Vec[S | T]: ...

    def __add__[V, S](
        self: Vec[V], value: IntoVec[V] | IntoVec[S], /
    ) -> Vec[V] | Vec[S | V]:
        match value:
            case Vec():
                return self.from_ref(self._inner + value._inner)  # pyright: ignore[reportArgumentType]
            case list():
                return self.from_ref(self._inner + value)  # pyright: ignore[reportArgumentType]

    @override
    def __iadd__(self, value: Iterable[T], /) -> Vec[T]:
        return self.from_ref(self._inner.__iadd__(value))

    def __mul__(self, value: SupportsIndex, /) -> Vec[T]:
        return Vec.from_ref(self._inner * value)

    def __rmul__(self, value: SupportsIndex, /) -> Vec[T]:
        return Vec.from_ref(value * self._inner)

    def __imul__(self, value: SupportsIndex, /) -> Vec[T]:
        self._inner *= value
        return self

    @override
    def __contains__(self, key: object, /) -> bool:
        return key in self._inner

    def __gt__(self, value: IntoVec[T], /) -> bool:
        match value:
            case Vec():
                return self._inner > value._inner
            case list():
                return self._inner > value

    def __ge__(self, value: IntoVec[T], /) -> bool:
        match value:
            case Vec():
                return self._inner >= value._inner
            case list():
                return self._inner >= value

    def __lt__(self, value: IntoVec[T], /) -> bool:
        match value:
            case Vec():
                return self._inner < value._inner
            case list():
                return self._inner < value

    def __le__(self, value: IntoVec[T], /) -> bool:
        match value:
            case Vec():
                return self._inner <= value._inner
            case list():
                return self._inner <= value

    @property
    @no_doctest
    def inner(self) -> list[T]:
        """The underlying `list` data structure.

        Useful when interoperating with functions that require a standard Python `list`.

        Returns:
            list[T]: The underlying list.
        """
        return self._inner

    @staticmethod
    def from_ref[V](data: list[V]) -> Vec[V]:
        """Create a `Vec` from a reference to an existing `list`.

        This method wraps the provided `list` without copying it, allowing for efficient creation of a `Vec`.

        This is the recommended way to create a `Vec` from foreign functions.

        Warning:
            Since the `Vec` directly references the original `list`, any modifications made to the `Vec` will also affect the original `list`, and vice versa.

        Args:
            data (list[V]): The `list` to wrap.

        Returns:
            Vec[V]: A new Vec instance wrapping the provided `list`.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> original_list = [1, 2, 3]
            >>> vec = Vec.from_ref(original_list)
            >>> vec
            Vec(1, 2, 3)
            >>> vec[0] = 10
            >>> original_list
            [10, 2, 3]

            ```
        """
        instance: Vec[V] = Vec.__new__(Vec)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @override
    def append(self, value: T) -> None:
        return self._inner.append(value)

    @override
    def extend(self, iterable: Iterable[T]) -> None:
        return self._inner.extend(iterable)

    @override
    def clear(self) -> None:
        return self._inner.clear()

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
            >>> from pyochain import Vec
            >>> Vec.from_ref([1, 2, 3]).repeat(2)
            Vec(1, 2, 3, 1, 2, 3)

            ```
        """
        return self.from_ref(self._inner * n)

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
            >>> vec = Vec.from_ref([1, 2, 3])
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
            >>> vec = Vec.from_ref(["a", "b", "c"])
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
            >>> Vec.from_ref([3, 1, 2]).sort()
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
            >>> Vec.from_ref(["3", "1", "2"]).sort_by(int)
            Vec('1', '2', '3')

            ```
        """
        self._inner.sort(key=key, reverse=reverse)
        return self

    def concat(self, other: list[T] | Self) -> Vec[T]:
        """Concatenate another `Vec` or `list` to **self** and return a new `Vec`.

        Note:
            This is equivalent to `list_1 + list_2` for standard lists.

        Args:
            other (list[T] | Self): The other `Vec` to concatenate.

        Returns:
            Vec[T]: The new `Vec` after concatenation.

        See Also:
            [`Vec::concat_mut`][concat_mut] which modifies **self** in place.

        Example:
            ```python
            >>> from pyochain import Vec
            >>> v1 = Vec.from_ref([1, 2, 3])
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
        return Vec.from_ref(data)

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
            >>> v1 = Vec.from_ref([1, 2, 3])
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
