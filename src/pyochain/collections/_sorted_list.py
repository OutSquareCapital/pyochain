# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from reprlib import recursive_repr
from typing import TYPE_CHECKING, Self, overload, override

from pyochain import Iter, Range, Vec
from pyochain.abc import PyoIterator, PyoMutableSequence
from pyochain.rs import InnerKeyLists, InnerLists, bisect_left, bisect_right

from ._base_sorted import BaseSortedList, KeyFunc, SortedCollection

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
        return self._inner.lists.iter().flatten()

    @override
    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return self._inner.lists.rev().flat_map(lambda x: x.rev())

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
        len_ = self._inner.len

        if len_ == 0:
            return Iter(())

        start, stop, _ = slice(start, stop).indices(self._inner.len)

        if start >= stop:
            return Iter(())

        min_pos, min_idx = self._pos(start)

        if stop == len_:
            max_pos = self._inner.lists.len() - 1
            max_idx = self._inner.lists[-1].len()
        else:
            max_pos, max_idx = self._pos(stop)

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

        if min_pos > max_pos:
            return Iter(())

        if min_pos == max_pos:
            if reverse:
                return Range(min_idx, max_idx).rev().map(lists[min_pos].__getitem__)

            return Range(min_idx, max_idx).iter().map(lists[min_pos].__getitem__)

        next_pos = min_pos + 1

        if next_pos == max_pos:
            if reverse:
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

            max_indices = Range(0, max_idx).iter().map(lists[max_pos].__getitem__)
            return (
                Range(min_idx, lists[min_pos].len())
                .iter()
                .map(lists[min_pos].__getitem__)
                .chain(
                    max_indices,
                )
            )

        if reverse:
            sublists = (
                Range(next_pos, max_pos)
                .rev()
                .map(lists.__getitem__)
                .flat_map(lambda x: x.rev())
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

        return (
            Range(min_idx, lists[min_pos].len())
            .iter()
            .map(lists[min_pos].__getitem__)
            .chain(
                Range(next_pos, max_pos).iter().flat_map(lists.__getitem__),
                Range(0, max_idx).iter().map(lists[max_pos].__getitem__),
            )
        )

    @override
    def irange(  # ruff:ignore[too-many-branches]
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        maxes = self._inner.maxes

        if maxes.is_empty():
            return Iter(())

        lists = self._inner.lists

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if minimum is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(maxes, minimum)

            if min_pos == maxes.len():
                return Iter(())

            min_idx = bisect_left(lists[min_pos], minimum)
        else:
            min_pos = bisect_right(maxes, minimum)

            if min_pos == maxes.len():
                return Iter(())

            min_idx = bisect_right(lists[min_pos], minimum)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if maximum is None:
            max_pos = maxes.len() - 1
            max_idx = lists[max_pos].len()
        elif inclusive[1]:
            max_pos = bisect_right(maxes, maximum)

            if max_pos == maxes.len():
                max_pos -= 1
                max_idx = lists[max_pos].len()
            else:
                max_idx = bisect_right(lists[max_pos], maximum)
        else:
            max_pos = bisect_left(maxes, maximum)

            if max_pos == maxes.len():
                max_pos -= 1
                max_idx = lists[max_pos].len()
            else:
                max_idx = bisect_left(lists[max_pos], maximum)

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


def identity[T](value: T) -> T:
    return value


class SortedKeyList[T, OT: SupportsRichComparison](SortedList[T]):  # pyright: ignore[reportInvalidTypeArguments]
    """Sorted-key list is a subtype of sorted list.

    The sorted-key list maintains values in comparison order based on the
    result of a key function applied to every value.

    All the same methods that are available in :class:`SortedList` are also
    available in :class:`SortedKeyList`.

    Additional methods provided:

    * :attr:`SortedKeyList.key`
    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Some examples below use:

    >>> from operator import neg
    >>> neg
    <built-in function neg>
    >>> neg(1)
    -1

    Optional `iterable` argument provides an initial iterable of values to
    initialize the sorted-key list.

    `key` argument defines a callable that, like the `key`
    argument to Python's `sorted` function, extracts a comparison key from
    each value. The default is the identity function.

    Runtime complexity: `O(n*log(n))`

    >>> from operator import neg
    >>> skl = SortedKeyList(key=neg)
    >>> skl
    SortedKeyList([], key=<built-in function neg>)
    >>> skl = SortedKeyList([3, 1, 2], key=neg)
    >>> skl
    SortedKeyList([3, 2, 1], key=<built-in function neg>)

    Args:
        iterable (Iterable[T] | None): initial values (optional)
        key (KeyFunc[T, OT]): function used to extract comparison key (optional)

    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        key: KeyFunc[T, OT] = identity,  # pyright: ignore[reportArgumentType]
    ) -> None:
        self._inner: InnerKeyLists[T, OT, OT] = InnerKeyLists(key)  # pyright: ignore[reportIncompatibleVariableOverride]

        if iterable is not None:
            self.update(iterable)

    @property
    @override
    def inner(self) -> InnerKeyLists[T, OT, OT]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Inner data structure for sorted-key list."""
        return self._inner

    @property
    def key(self) -> KeyFunc[T, OT]:
        """Function used to extract comparison key from values."""
        return self._inner.key

    @override
    def clear(self) -> None:
        self._inner.clear()

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        min_key = self._inner.key(minimum) if minimum is not None else None
        max_key = self._inner.key(maximum) if maximum is not None else None
        return self.irange_key(min_key, max_key, inclusive, reverse=reverse)

    def irange_key(  # ruff:ignore[too-many-branches]
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        """Create an iterator of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange_key(-14, -12)
        >>> list(it)
        [14, 13, 12]

        Args:
            min_key (OT | None): minimum key to start iterating
            max_key (OT | None): maximum key to stop iterating
            inclusive (tuple[bool, bool]): pair of booleans
            reverse (bool): yield values in reverse order

        Returns:
            Iterator[T]: iterator of values between `min_key` and `max_key`

        """
        maxes = self._inner.maxes

        if maxes.is_empty():
            return Iter(())

        keys = self._inner.keys

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if min_key is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(maxes, min_key)

            if min_pos == maxes.len():
                return Iter(())

            min_idx = bisect_left(keys[min_pos], min_key)
        else:
            min_pos = bisect_right(maxes, min_key)

            if min_pos == maxes.len():
                return Iter(())

            min_idx = bisect_right(keys[min_pos], min_key)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if max_key is None:
            max_pos = maxes.len() - 1
            max_idx = keys[max_pos].len()
        elif inclusive[1]:
            max_pos = bisect_right(maxes, max_key)

            if max_pos == maxes.len():
                max_pos -= 1
                max_idx = keys[max_pos].len()
            else:
                max_idx = bisect_right(keys[max_pos], max_key)
        else:
            max_pos = bisect_left(maxes, max_key)

            if max_pos == maxes.len():
                max_pos -= 1
                max_idx = keys[max_pos].len()
            else:
                max_idx = bisect_left(keys[max_pos], max_key)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse=reverse)

    def bisect_key_left(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        If the `key` is already present, the insertion point will be before (to
        the left of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_left(-1)
        4

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        return self._inner.bisect_key_left(key)

    def bisect_key_right(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        Similar to `bisect_key_left`, but if `key` is already present, the
        insertion point will be after (to the right of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_right(-1)
        5

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        return self._inner.bisect_key_right(key)

    @override
    def copy(self) -> Self:
        return self.__class__(self, key=self._inner.key)

    @override
    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted-key list containing all values in both sequences.

        ``skl.__add__(other)`` <==> ``skl + other``

        Values in `other` do not need to be in sorted-key order.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl1 = SortedKeyList([5, 4, 3], key=neg)
        >>> skl2 = SortedKeyList([2, 1, 0], key=neg)
        >>> skl1 + skl2
        SortedKeyList([5, 4, 3, 2, 1, 0], key=<built-in function neg>)

        :param other: other iterable
        :return: new sorted-key list

        """
        values = self._inner.collapse_lists()
        values.extend(other)
        return self.__class__(values, key=self._inner.key)

    @override
    def __mul__(self, num: int) -> Self:
        """Return new sorted-key list with `num` shallow copies of values.

        ``skl.__mul__(num)`` <==> ``skl * num``

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([3, 2, 1], key=neg)
        >>> skl * 2
        SortedKeyList([3, 3, 2, 2, 1, 1], key=<built-in function neg>)

        :param int num: count of shallow copies
        :return: new sorted-key list

        """
        values = self._inner.collapse_lists().repeat(num)
        return self.__class__(values, key=self._inner.key)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T], KeyFunc[T, OT]]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        values = self._inner.collapse_lists()
        return (self.__class__, (values, self.key))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted-key list.

        ``skl.__repr__()`` <==> ``repr(skl)``

        :return: string representation

        """
        type_name = self.__class__.__name__
        return f"{type_name}({list(self)!r}, key={self._inner.key!r})"
