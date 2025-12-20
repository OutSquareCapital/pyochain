from __future__ import annotations

from collections.abc import Iterable, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, overload

import cytoolz as cz

from ._common import CommonMethods, convert_data

if TYPE_CHECKING:
    from ._lazy import Iter


class Seq[T](CommonMethods[T], Sequence[T]):
    """`Seq` represent an in memory Sequence.

    Implements the `Sequence` Protocol from `collections.abc`, so it can be used as a standard immutable sequence.

    Provides a subset of `Iter` methods with eager evaluation, and is the return type of `Iter.collect()`.

    The underlying data structure is an immutable tuple, hence the memory efficiency is better than a `Vec`.

    You can create a `Seq` from any `Iterable` (like a list, or polars.Series) or unpacked values using the `from_` class method.

    If you already have a tuple, simply pass it to the constructor, without runtime checks.

    Args:
            data (tuple[T, ...]): The data to initialize the Seq with.
    """

    _inner: tuple[T, ...]

    __slots__ = ("_inner",)

    def __init__(self, data: tuple[T, ...]) -> None:
        self._inner = data  # pyright: ignore[reportIncompatibleVariableOverride]

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    def __getitem__(self, index: int | slice[Any, Any, Any]) -> T | Sequence[T]:
        return self._inner.__getitem__(index)

    def __len__(self) -> int:
        return len(self._inner)

    @overload
    @staticmethod
    def from_[U](data: Iterable[U]) -> Seq[U]: ...
    @overload
    @staticmethod
    def from_[U](data: U, *more_data: U) -> Seq[U]: ...
    @staticmethod
    def from_[U](data: Iterable[U] | U, *more_data: U) -> Seq[U]:
        """Create a `Seq` from an `Iterable` or unpacked values.

        Prefer using the standard constructor, as this method involves extra checks and conversions steps.

        Args:
            data (Iterable[U] | U): Iterable to convert into a sequence, or a single value.
            *more_data (U): Unpacked items to include in the sequence, if 'data' is not an Iterable.

        Returns:
            Seq[U]: A new Seq instance containing the provided data.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq.from_(1, 2, 3)
        Seq(1, 2, 3)

        ```
        """
        converted = convert_data(data, *more_data)
        return Seq(converted if isinstance(converted, tuple) else tuple(converted))

    def iter(self) -> Iter[T]:
        """Get an iterator over the sequence.

        Call this to switch to lazy evaluation.

        Returns:
            Iter[T]: An `Iter` instance wrapping an iterator over the sequence.
        """
        return self._lazy(iter)

    def is_distinct(self) -> bool:
        """Return True if all items are distinct.

        Returns:
            bool: True if all items are distinct, False otherwise.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).is_distinct()
        True

        ```
        """
        return self.into(cz.itertoolz.isdistinct)


class Vec[T](Seq[T], MutableSequence[T]):
    """A mutable sequence wrapper with functional API.

    Implement `MutableSequence` Protocol from `collections.abc` so it can be used as a standard mutable sequence.

    Unlike `Seq` which is immutable, `Vec` allows in-place modification of elements.

    Implement the `MutableSequence` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable sequence.

    If you already have a list, simply pass it to the constructor, without runtime checks.

    Otherwise, use the `from_` class method to create a `Vec` from any `Iterable` or unpacked values.

    Args:
        data (list[T]): The mutable sequence to wrap.
    """

    _inner: list[T]

    def __init__(self, data: list[T]) -> None:
        self._inner = data  # type: ignore[override]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        return self._inner.__setitem__(index, value)  # type: ignore[arg-type]

    def __delitem__(self, index: int | slice) -> None:
        self._inner.__delitem__(index)

    def insert(self, index: int, value: T) -> None:
        """Inserts an element at position index within the vector, shifting all elements after it to the right.

        Args:
            index (int): Position where to insert the element.
            value (T): The element to insert.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> vec = pc.Vec(['a', 'b', 'c'])
        >>> vec.insert(1, 'd')
        >>> vec
        Vec('a', 'd', 'b', 'c')
        >>> vec.insert(4, 'e')
        >>> vec
        Vec('a', 'd', 'b', 'c', 'e')

        ```
        """
        self._inner.insert(index, value)

    @overload
    @staticmethod
    def from_[U](data: Iterable[U]) -> Vec[U]: ...
    @overload
    @staticmethod
    def from_[U](data: U, *more_data: U) -> Vec[U]: ...
    @staticmethod
    def from_[U](data: Iterable[U] | U, *more_data: U) -> Vec[U]:
        """Create a `Vec` from an `Iterable` or unpacked values.

        Prefer using the standard constructor, as this method involves extra checks and conversions steps.

        Args:
            data (Iterable[U] | U): Iterable to convert into a sequence, or a single value.
            *more_data (U): Unpacked items to include in the sequence, if 'data' is not an Iterable.

        Returns:
            Vec[U]: A new Vec instance containing the provided data.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Vec.from_(1, 2, 3)
        Vec(1, 2, 3)

        ```
        """
        converted = convert_data(data, *more_data)
        return Vec(converted if isinstance(converted, list) else list(converted))

    @staticmethod
    def new() -> Vec[T]:
        """Create an empty `Vec`.

        Make sure to specify the type when calling this method, e.g., `Vec[int].new()`.

        Otherwise, `T` will be inferred as `Any`.

        Returns:
            Vec[T]: A new empty Vec instance.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Vec.new()
        Vec()

        ```
        """
        return Vec([])
