from __future__ import annotations

from collections.abc import Collection, Iterable, MutableSequence, Sequence
from typing import Any, Self, overload

import cytoolz as cz

from ._iter import CommonMethods, convert_data


class Set[T](CommonMethods[T], Collection[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `Collection` Protocol from `collections.abc`, so it can be used as a standard immutable collection.

    Provides a subset of `Iter` methods with eager evaluation, and is returned from some `Iter/Seq/Vec` methods.

    The underlying data structure is a `set` or `frozenset`.

    You can create a `Set` from any `Iterable` (like a list, or polars.Series) or unpacked values using the `from_` class method.

    If you already have a `set` or `frozenset`, simply pass it to the constructor, without runtime checks.

    Args:
            data (set[T] | frozenset[T]): The data to initialize the Set with.
    """

    _inner: set[T] | frozenset[T]

    __slots__ = ("_inner",)

    def __init__(self, data: set[T] | frozenset[T]) -> None:
        self._inner = data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __contains__(self, item: object) -> bool:
        return self._inner.__contains__(item)

    def __len__(self) -> int:
        return len(self._inner)

    @overload
    @staticmethod
    def from_[U](data: Iterable[U]) -> Set[U]: ...
    @overload
    @staticmethod
    def from_[U](data: U, *more_data: U) -> Set[U]: ...
    @staticmethod
    def from_[U](data: Iterable[U] | U, *more_data: U) -> Set[U]:
        """Create a `Set` from an `Iterable` or unpacked values.

        Prefer using the standard constructor, as this method involves extra checks and conversions steps.

        The underlying data structure will be a `frozenset`.

        Args:
            data (Iterable[U] | U): Iterable to convert into a sequence, or a single value.
            *more_data (U): Unpacked items to include in the sequence, if 'data' is not an Iterable.

        Returns:
            Set[U]: A new Set instance containing the provided data.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Set.from_(1, 2, 3)
        Set(1, 2, 3)

        ```
        """
        converted = convert_data(data, *more_data)
        return Set(
            converted
            if isinstance(converted, (set, frozenset))
            else frozenset(converted)
        )

    def union(self, *others: Iterable[T]) -> Set[T]:
        """Return the union of this iterable and 'others'.

        Note:
            This method consumes inner data and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to include in the union.

        Returns:
            Set[T]: A new `Set` containing the union of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).union([2, 3], [4]).iter().sort()
        Vec(1, 2, 3, 4)

        ```
        """
        return self.__class__(self._inner.union(*others))

    def intersection(self, *others: Iterable[Any]) -> Set[T]:
        """Return the elements common to this iterable and 'others'.

        Is the opposite of `difference`.

        See Also:
            - `difference`
            - `diff_symmetric`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[Any]): Other iterables to intersect with.

        Returns:
            Set[T]: A new `Set` containing the intersection of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).intersection([2, 3], [2])
        Set(2,)

        ```
        """
        return self.__class__(self._inner.intersection(*others))

    def difference(self, *others: Iterable[T]) -> Set[T]:
        """Return the difference of this iterable and 'others'.

        See Also:
            - `intersection`
            - `diff_symmetric`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to subtract from this iterable.

        Returns:
            Set[T]: A new `Set` containing the difference of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).difference([2, 3])
        Set(1,)

        ```
        """
        return self.__class__(self._inner.difference(*others))

    def symmetric_difference(self, *others: Iterable[T]) -> Set[T]:
        """Return the symmetric difference (XOR) of this iterable and 'others'.

        (Elements in either 'self' or 'others' but not in both).

        **See Also**:
            - `intersection`
            - `difference`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to compute the symmetric difference with.

        Returns:
            Set[T]: A new `Set` containing the symmetric difference of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).symmetric_difference([2, 3]).iter().sort()
        Vec(1, 3)
        >>> pc.Set({1, 2, 3}).symmetric_difference([3, 4, 5]).iter().sort()
        Vec(1, 2, 4, 5)

        ```
        """
        return self.__class__(self._inner.symmetric_difference(*others))

    def is_subset(self, other: Iterable[Any]) -> bool:
        """Test whether every element in the set is in **other**.

        Args:
            other (Iterable[Any]): Another iterable to compare with.

        Returns:
            bool: True if this set is a subset of **other**, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2}).is_subset([1, 2, 3])
        True
        >>> pc.Set({1, 4}).is_subset([1, 2, 3])
        False

        ```
        """
        return self._inner.issubset(other)

    def is_superset(self, other: Iterable[Any]) -> bool:
        """Test whether every element in **other** is in the set.

        Args:
            other (Iterable[Any]): Another iterable to compare with.

        Returns:
            bool: True if this set is a superset of **other**, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 3}).is_superset([1, 2])
        True
        >>> pc.Set({1, 2}).is_superset([1, 2, 3])
        False

        ```
        """
        return self._inner.issuperset(other)

    def is_disjoint(self, other: Iterable[Any]) -> bool:
        """Test whether the set and **other** have no elements in common.

        Args:
            other (Iterable[Any]): Another iterable to compare with.

        Returns:
            bool: True if the sets have no elements in common, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2}).is_disjoint([3, 4])
        True
        >>> pc.Set({1, 2}).is_disjoint([2, 3])
        False

        ```
        """
        return self._inner.isdisjoint(other)


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

    def __len__(self) -> int:
        return len(self._inner)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    def __getitem__(self, index: int | slice[Any, Any, Any]) -> T | Sequence[T]:
        return self._inner.__getitem__(index)

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
    __slots__ = ("_inner",)

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

    @classmethod
    def new(cls) -> Self:
        """Create an empty `Vec`.

        Make sure to specify the type when calling this method, e.g., `Vec[int].new()`.

        Otherwise, `T` will be inferred as `Any`.

        Returns:
            Self: A new empty Vec instance.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Vec.new()
        Vec()

        ```
        """
        return cls([])
