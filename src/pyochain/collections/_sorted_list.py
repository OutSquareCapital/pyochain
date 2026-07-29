# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from enum import Enum
from reprlib import recursive_repr
from typing import TYPE_CHECKING, Self, overload, override

from pyochain import Iter, Range, Vec
from pyochain.abc import PyoIterator, PyoMutableSequence
from pyochain.rs import InnerLists

from ._base_sorted import BaseSortedList, SortedCollection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import SupportsRichComparison


class SortedList[T: SupportsRichComparison](
    BaseSortedList[T], SortedCollection[T], PyoMutableSequence[T]
):
    """Sorted list is a sorted mutable sequence.

    Sorted list values are maintained in sorted order.

    Sorted list values must be comparable. The total ordering of values must
    not change while they are stored in the sorted list.

    Methods for adding values:

    * :func:`SortedList.add`
    * :func:`SortedList.update`
    * :func:`SortedList.__add__`
    * :func:`SortedList.__iadd__`
    * :func:`SortedList.__mul__`
    * :func:`SortedList.__imul__`

    Methods for removing values:

    * :func:`SortedList.clear`
    * :func:`SortedList.discard`
    * :func:`SortedList.remove`
    * :func:`SortedList.pop`
    * :func:`SortedList.__delitem__`

    Methods for looking up values:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.__contains__`
    * :func:`SortedList.__getitem__`

    Methods for iterating values:

    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.__iter__`
    * :func:`SortedList.__reversed__`

    Methods for miscellany:

    * :func:`SortedList.copy`
    * :func:`SortedList.__len__`
    * :func:`SortedList.__repr__`
    * :func:`SortedList._check`
    * :func:`SortedList.reset`

    Sorted lists use lexicographical ordering semantics when compared to other
    sequences.

    Some methods of mutable sequences are not supported and will raise
    not-implemented error.

    Optional `iterable` argument provides an initial iterable of values to
    initialize the sorted list.

    Runtime complexity: `O(n*log(n))`

    >>> sl = SortedList()
    >>> sl
    SortedList([])
    >>> sl = SortedList([3, 1, 2, 5, 4])
    >>> sl
    SortedList([1, 2, 3, 4, 5])

    """

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._inner: InnerLists[T, T] = InnerLists[T, T]()  # pyright: ignore[reportIncompatibleVariableOverride]

        if iterable is not None:
            self._inner.update(iterable)

    @property
    def inner(self) -> InnerLists[T, T]:
        return self._inner

    @override
    def clear(self) -> None:
        self._inner.clear()

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        """Raise not-implemented error.

        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``

        :raises NotImplementedError: use ``del sl[index]`` and
            ``sl.add(value)`` instead

        """
        message = "use ``del sl[index]`` and ``sl.add(value)`` instead"
        raise NotImplementedError(message)

    @override
    def __iter__(self) -> PyoIterator[T]:
        """Return an iterator over the sorted list.

        ``sl.__iter__()`` <==> ``iter(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return Iter(self._inner.lists).flatten()

    @override
    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return Iter(reversed(self._inner.lists)).flat_map(reversed)

    @override
    def reverse(self) -> None:
        """Raise not-implemented error.

        Sorted list maintains values in ascending sort order. Values may not be
        reversed in-place.

        Use ``reversed(sl)`` for an iterator over values in descending sort
        order.

        Implemented to override `MutableSequence.reverse` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``reversed(sl)`` instead

        """
        msg = "use ``reversed(sl)`` instead"
        raise NotImplementedError(msg)

    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        match self._inner.islice(start, stop):
            case None:
                return Iter(())
            case (min_pos, min_idx, max_pos, max_idx):
                return self._islice(min_pos, min_idx, max_pos, max_idx, reverse=reverse)

    def _islice(  # ruff:ignore[too-many-return-statements]
        self, min_pos: int, min_idx: int, max_pos: int, max_idx: int, *, reverse: bool
    ) -> PyoIterator[T]:
        """Return an iterator that slices sorted list using two index pairs.

        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the
        first inclusive and the latter exclusive. See `_pos` for details on how
        an index is converted to an index pair.

        When `reverse` is `True`, values are yielded from the iterator in
        reverse order.

        """
        lists = self._inner.lists
        kind = SliceKind.new(min_pos, max_pos, reverse=reverse)

        next_pos = min_pos + 1
        match kind:
            case SliceKind.Empty:
                return Iter(())
            case SliceKind.MinEqMax:
                return Range(min_idx, max_idx).iter().map(lists[min_pos].__getitem__)
            case SliceKind.MinEqMaxRev:
                return Range(min_idx, max_idx).rev().map(lists[min_pos].__getitem__)
            case SliceKind.NextEqMax:
                max_indices = Range(0, max_idx).iter().map(lists[max_pos].__getitem__)
                return (
                    Range(min_idx, lists[min_pos].len())
                    .iter()
                    .map(lists[min_pos].__getitem__)
                    .chain(
                        max_indices,
                    )
                )
            case SliceKind.NextEqMaxRev:
                min_indices = (
                    Range(min_idx, lists[min_pos].len())
                    .rev()
                    .map(lists[min_pos].__getitem__)
                )
                return (
                    Range(0, max_idx)
                    .rev()
                    .map(lists[max_pos].__getitem__)
                    .chain(min_indices)
                )
            case SliceKind.MinLtMax:
                return (
                    Range(min_idx, lists[min_pos].len())
                    .iter()
                    .map(lists[min_pos].__getitem__)
                    .chain(
                        Range(next_pos, max_pos).iter().flat_map(lists.__getitem__),
                        Range(0, max_idx).iter().map(lists[max_pos].__getitem__),
                    )
                )
            case SliceKind.MinLtMaxRev:
                sublists = (
                    Range(next_pos, max_pos)
                    .rev()
                    .map(lists.__getitem__)
                    .flat_map(reversed)
                )
                return (
                    Range(0, max_idx)
                    .rev()
                    .map(lists[max_pos].__getitem__)
                    .chain(
                        sublists,
                        Range(min_idx, lists[min_pos].len())
                        .rev()
                        .map(lists[min_pos].__getitem__),
                    )
                )

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        match self._inner.irange(minimum, maximum, inclusive=inclusive):
            case None:
                return Iter(())
            case (min_pos, min_idx, max_pos, max_idx):
                return self._islice(min_pos, min_idx, max_pos, max_idx, reverse=reverse)

    @override
    def __len__(self) -> int:
        """Return the size of the sorted list.

        ``sl.__len__()`` <==> ``len(sl)``

        :return: size of sorted list

        """
        return self._inner.len

    @override
    def copy(self) -> Self:
        return self.__class__(self)

    def __copy__(self) -> Self:
        return self.copy()

    @override
    def append(self, value: T) -> None:
        """Raise not-implemented error.

        Implemented to override `MutableSequence.append` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def extend(self, values: object) -> None:
        """Raise not-implemented error.

        Implemented to override `MutableSequence.extend` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.update(values)`` instead

        """
        msg = "use ``sl.update(values)`` instead"
        raise NotImplementedError(msg)

    @override
    def insert(self, index: int, value: T) -> None:
        """Raise not-implemented error.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted list containing all values in both sequences.

        ``sl.__add__(other)`` <==> ``sl + other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(n*log(n))`

        >>> sl1 = SortedList("bat")
        >>> sl2 = SortedList("cat")
        >>> sl1 + sl2
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        :param other: other iterable
        :return: new sorted list

        """
        values = self._inner.collapse_lists()
        values.extend(other)
        return self.__class__(values)

    def __radd__(self, other: Iterable[T]) -> Self:
        return self.__add__(other)

    @override
    def __iadd__(self, other: Iterable[T]) -> Self:
        """Update sorted list with values from `other`.

        ``sl.__iadd__(other)`` <==> ``sl += other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> sl = SortedList("bat")
        >>> sl += "cat"
        >>> sl
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        Args:
            other (Iterable[T]): other iterable

        Returns:
            Self: existing sorted list

        """
        self.update(other)
        return self

    @override
    def __mul__(self, num: int) -> Self:
        """Return new sorted list with `num` shallow copies of values.

        ``sl.__mul__(num)`` <==> ``sl * num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList("abc")
        >>> sl * 3
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        :param int num: count of shallow copies
        :return: new sorted list

        """
        values = self._inner.collapse_lists().repeat(num)
        return self.__class__(values)

    def __rmul__(self, num: int) -> Self:
        return self.__mul__(num)

    def __imul__(self, num: int) -> Self:
        """Update the sorted list with `num` shallow copies of values.

        ``sl.__imul__(num)`` <==> ``sl *= num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList("abc")
        >>> sl *= 3
        >>> sl
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        Args:
            num (int): count of shallow copies

        Returns:
            Self: existing sorted list

        """
        values = self._inner.collapse_lists().repeat(num)
        self.clear()
        self.update(values)
        return self

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T]]]:
        values = self._inner.collapse_lists()
        return (self.__class__, (values,))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted list.

        ``sl.__repr__()`` <==> ``repr(sl)``

        :return: string representation

        """
        return f"{self.__class__.__name__}({list(self)!r})"


class SliceKind(Enum):
    Empty = 1
    MinEqMax = 2
    MinEqMaxRev = 3
    NextEqMax = 4
    NextEqMaxRev = 5
    MinLtMax = 6
    MinLtMaxRev = 7

    @classmethod
    def new(cls, min_pos: int, max_pos: int, *, reverse: bool) -> SliceKind:  # ruff:ignore[too-many-return-statements]
        next_pos = min_pos + 1
        if min_pos > max_pos:
            return cls.Empty

        if min_pos == max_pos:
            if reverse:
                return cls.MinEqMaxRev
            return cls.MinEqMax

        if next_pos == max_pos:
            if reverse:
                return cls.NextEqMaxRev
            return cls.NextEqMax

        if reverse:
            return cls.MinLtMaxRev

        return cls.MinLtMax
