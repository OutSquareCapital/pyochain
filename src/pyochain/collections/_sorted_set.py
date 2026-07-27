# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from reprlib import recursive_repr
from typing import TYPE_CHECKING, Any, Self, overload, override

from pyochain import Iter, Set
from pyochain.abc import PyoIterator, PyoMutableSet, PyoSequence

from ._base_sorted import BaseSortedListSet, KeyFunc, SortedCollection
from ._sorted_list import SortedKeyList, SortedList, identity

if TYPE_CHECKING:
    from types import NotImplementedType

    from pyochain import Vec
    from pyochain._types import SupportsHashableAndRichComparison

    type SetKeyFunc[T, OT: SupportsHashableAndRichComparison] = KeyFunc[T, OT]


class SortedSet[T: SupportsHashableAndRichComparison](  # ruff:ignore[eq-without-hash]
    PyoMutableSet[T], PyoSequence[T], SortedCollection[T], BaseSortedListSet[T]
):
    """Sorted set is a sorted mutable set.

    Sorted set values are maintained in sorted order. The design of sorted set
    is simple: sorted set uses a set for set-operations and maintains a sorted
    list of values.

    Sorted set values must be hashable and comparable. The hash and total
    ordering of values must not change while they are stored in the sorted set.

    Mutable set methods:

    * :func:`SortedSet.__contains__`
    * :func:`SortedSet.__iter__`
    * :func:`SortedSet.__len__`
    * :func:`SortedSet.add`
    * :func:`SortedSet.discard`

    Sequence methods:

    * :func:`SortedSet.__getitem__`
    * :func:`SortedSet.__delitem__`
    * :func:`SortedSet.__reversed__`

    Methods for removing values:

    * :func:`SortedSet.clear`
    * :func:`SortedSet.pop`
    * :func:`SortedSet.remove`

    Set-operation methods:

    * :func:`SortedSet.difference`
    * :func:`SortedSet.difference_update`
    * :func:`SortedSet.intersection`
    * :func:`SortedSet.intersection_update`
    * :func:`SortedSet.symmetric_difference`
    * :func:`SortedSet.symmetric_difference_update`
    * :func:`SortedSet.union`
    * :func:`SortedSet.update`

    Methods for miscellany:

    * :func:`SortedSet.copy`
    * :func:`SortedSet.count`
    * :func:`SortedSet.__repr__`
    * :func:`SortedSet._check`

    Sorted list methods available:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted set comparisons use subset and superset relations. Two sorted sets
    are equal if and only if every element of each sorted set is contained in
    the other (each is a subset of the other). A sorted set is less than
    another sorted set if and only if the first sorted set is a proper subset
    of the second sorted set (is a subset, but is not equal). A sorted set is
    greater than another sorted set if and only if the first sorted set is a
    proper superset of the second sorted set (is a superset, but is not equal).

    Optional `iterable` argument provides an initial iterable of values to
    initialize the sorted set.

    Runtime complexity: `O(n*log(n))`

    >>> ss = SortedSet([3, 1, 2, 5, 4])
    >>> ss
    SortedSet([1, 2, 3, 4, 5])

    Args:
        iterable (Iterable[T] | None): initial values (optional)

    """

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        # SortedSet._fromset calls SortedSet.__init__ after initializing the
        # _set attribute. So only create a new set if the _set attribute is not
        # already present.

        if not hasattr(self, "_set"):
            self._set: set[T] = set[T]()

        self._list: SortedList[T] = SortedList(self._set)

        if iterable is not None:
            _ = self.update(iterable)

    @override
    def reset(self, load: int) -> None:
        return self._list.reset(load)

    @override
    def bisect_left(self, value: T) -> int:
        return self._list.bisect_left(value)

    @override
    def bisect_right(self, value: T) -> int:
        return self._list.bisect_right(value)

    @override
    def index(self, value: T, start: int | None = None, stop: int | None = None) -> int:
        return self._list.index(value, start, stop)

    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        return self._list.islice(start, stop, reverse=reverse)

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        return self._list.irange(minimum, maximum, inclusive, reverse=reverse)

    @override
    def is_disjoint(self, other: Iterable[object]) -> bool:
        return self._set.isdisjoint(other)

    @override
    def is_subset(self, other: Iterable[object]) -> bool:
        return self._set.issubset(other)

    @override
    def is_superset(self, other: Iterable[object]) -> bool:
        return self._set.issuperset(other)

    def _fromset(self, values: set[T]) -> Self:
        sorted_set = self.__new__(self.__class__)
        sorted_set._set = values
        sorted_set.__init__()  # ruff:ignore[unnecessary-dunder-call]
        return sorted_set

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted set.

        ``ss.__contains__(value)`` <==> ``value in ss``

        Runtime complexity: `O(1)`

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> 3 in ss
        True

        :param value: search for value in sorted set
        :return: true if `value` in sorted set

        """
        return value in self._set

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | Vec[T]:
        """Lookup value at `index` in sorted set.

        ``ss.__getitem__(index)`` <==> ``ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet("abcde")
        >>> ss[2]
        'c'
        >>> ss[-1]
        'e'
        >>> ss[2:5]
        Vec('c', 'd', 'e')

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            T | Vec[T]: value or list of values

        """
        return self._list[index]

    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted set.

        ``ss.__delitem__(index)`` <==> ``del ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet("abcde")
        >>> del ss[2]
        >>> ss
        SortedSet(['a', 'b', 'd', 'e'])
        >>> del ss[:2]
        >>> ss
        SortedSet(['d', 'e'])

        :param index: integer or slice for indexing
        :raises IndexError: if index out of range

        """
        set_ = self._set
        list_ = self._list
        match index:
            case slice():
                values = list_[index]
                set_.difference_update(values)
            case int():
                value = list_[index]
                set_.remove(value)
        del list_[index]

    @override
    def __eq__(self, other: object) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set == other._set
            case AbstractSet():
                return self._set == other
            case _:
                return NotImplemented

    @override
    def __ne__(self, other: object) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set != other._set
            case AbstractSet():
                return self._set != other
            case _:
                return NotImplemented

    @override
    def __lt__(
        self, other: AbstractSet[object] | SortedSet[T] | object
    ) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set < other._set
            case AbstractSet():
                return self._set < other
            case _:
                return NotImplemented

    @override
    def __gt__(
        self, other: AbstractSet[object] | SortedSet[T] | object
    ) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set > other._set
            case AbstractSet():
                return self._set > other
            case _:
                return NotImplemented

    @override
    def __le__(
        self, other: AbstractSet[object] | SortedSet[T] | object
    ) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set <= other._set
            case AbstractSet():
                return self._set <= other
            case _:
                return NotImplemented

    @override
    def __ge__(
        self, other: AbstractSet[object] | SortedSet[T] | object
    ) -> bool | NotImplementedType:
        match other:
            case SortedSet():
                return self._set >= other._set
            case AbstractSet():
                return self._set >= other
            case _:
                return NotImplemented

    @override
    def __len__(self) -> int:
        """Return the size of the sorted set.

        ``ss.__len__()`` <==> ``len(ss)``

        :return: size of sorted set

        """
        return len(self._set)

    @override
    def __iter__(self) -> PyoIterator[T]:
        """Return an iterator over the sorted set.

        ``ss.__iter__()`` <==> ``iter(ss)``

        Iterating the sorted set while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return self._list.iter()

    @override
    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted set.

        ``ss.__reversed__()`` <==> ``reversed(ss)``

        Iterating the sorted set while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return self._list.rev()

    @override
    def add(self, value: T) -> None:
        """Add `value` to sorted set.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet()
        >>> ss.add(3)
        >>> ss.add(1)
        >>> ss.add(2)
        >>> ss
        SortedSet([1, 2, 3])

        :param value: value to add to sorted set

        """
        set_ = self._set
        if value not in set_:
            set_.add(value)
            self._list.add(value)

    @override
    def clear(self) -> None:
        """Remove all values from sorted set.

        Runtime complexity: `O(n)`

        """
        self._set.clear()
        self._list.clear()

    @override
    def copy(self) -> Self:
        """Return a shallow copy of the sorted set.

        Runtime complexity: `O(n)`

        :return: new sorted set

        """
        return self._fromset(set(self._set))

    def __copy__(self) -> Self:
        return self.copy()

    @override
    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted set.

        Runtime complexity: `O(1)`

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.count(3)
        1

        :param value: value to count in sorted set
        :return: count

        """
        return 1 if value in self._set else 0

    @override
    def discard(self, value: T) -> None:
        """Remove `value` from sorted set if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.discard(5)
        >>> ss.discard(0)
        >>> ss == set([1, 2, 3, 4])
        True

        :param value: `value` to discard from sorted set

        """
        set_ = self._set
        if value in set_:
            set_.remove(value)
            self._list.remove(value)

    @override
    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted set.

        Raise :exc:`IndexError` if the sorted set is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet("abcde")
        >>> ss.pop()
        'e'
        >>> ss.pop(2)
        'c'
        >>> ss
        SortedSet(['a', 'b', 'd'])

        Args:
            index (int): index of value (default -1)

        Returns:
            T: value at `index`

        """
        # pylint: disable=arguments-differ
        value = self._list.pop(index)
        self._set.remove(value)
        return value

    @override
    def remove(self, value: T) -> None:
        """Remove `value` from sorted set; `value` must be a member.

        If `value` is not a member, raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.remove(5)
        >>> ss == set([1, 2, 3, 4])
        True
        >>> ss.remove(0)
        Traceback (most recent call last):
          ...
        KeyError: 0

        :param value: `value` to remove from sorted set
        :raises KeyError: if `value` is not in sorted set

        """
        self._set.remove(value)
        self._list.remove(value)

    @override
    def difference(self, *iterables: Iterable[Any]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the difference of two or more sets as a new sorted set.

        The `difference` method also corresponds to operator ``-``.

        ``ss.__sub__(iterable)`` <==> ``ss - iterable``

        The difference is all values that are in this sorted set but not the
        other `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.difference([4, 5, 6, 7])
        SortedSet([1, 2, 3])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        diff = self._set.difference(*iterables)
        return self._fromset(diff)

    @override
    def __sub__(self, other: Iterable[Any]) -> Self:
        return self.difference(other)

    def difference_update(self, *iterables: Iterable[T]) -> Self:
        """Remove all values of `iterables` from this sorted set.

        The `difference_update` method also corresponds to operator ``-=``.

        ``ss.__isub__(iterable)`` <==> ``ss -= iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.difference_update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3])

        Args:
            *iterables (Iterable[T]): iterable arguments

        Returns:
            Self: updated sorted set

        """
        set_ = self._set
        list_ = self._list
        values = Iter(iterables).flatten().collect(Set)
        if (4 * values.len()) > len(set_):
            set_.difference_update(values)
            list_.clear()
            list_.update(set_)
        else:
            for value in values:
                self.discard(value)
        return self

    @override
    def __isub__(self, other: Iterable[Any]) -> Self:
        return self.difference_update(other)

    @override
    def intersection(self, *iterables: Iterable[Any]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the intersection of two or more sets as a new sorted set.

        The `intersection` method also corresponds to operator ``&``.

        ``ss.__and__(iterable)`` <==> ``ss & iterable``

        The intersection is all values that are in this sorted set and each of
        the other `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.intersection([4, 5, 6, 7])
        SortedSet([4, 5])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        intersect = self._set.intersection(*iterables)
        return self._fromset(intersect)

    @override
    def __and__(self, other: Iterable[Any]) -> Self:
        return self.intersection(other)

    def __rand__(self, other: Iterable[Any]) -> Self:
        return self.intersection(other)

    def intersection_update(self, *iterables: Iterable[Any]) -> Self:
        """Update the sorted set with the intersection of `iterables`.

        The `intersection_update` method also corresponds to operator ``&=``.

        ``ss.__iand__(iterable)`` <==> ``ss &= iterable``

        Keep only values found in itself and all `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.intersection_update([4, 5, 6, 7])
        >>> ss
        SortedSet([4, 5])

        Args:
            *iterables (Iterable[Any]): iterable arguments

        Returns:
            Self: updated sorted set

        """
        self._set.intersection_update(*iterables)
        self._list.clear()
        self._list.update(self._set)
        return self

    @override
    def __iand__(self, other: Iterable[Any]) -> Self:
        return self.intersection_update(other)

    @override
    def symmetric_difference(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the symmetric difference with `other` as a new sorted set.

        The `symmetric_difference` method also corresponds to operator ``^``.

        ``ss.__xor__(other)`` <==> ``ss ^ other``

        The symmetric difference is all values tha are in exactly one of the
        sets.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.symmetric_difference([4, 5, 6, 7])
        SortedSet([1, 2, 3, 6, 7])

        :param other: `other` iterable
        :return: new sorted set

        """
        diff = self._set.symmetric_difference(other)
        return self._fromset(diff)

    @override
    def __xor__(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.symmetric_difference(other)

    def __rxor__(self, other: Iterable[T]) -> Self:
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, other: Iterable[T]) -> Self:
        """Update the sorted set with the symmetric difference with `other`.

        The `symmetric_difference_update` method also corresponds to operator
        ``^=``.

        ``ss.__ixor__(other)`` <==> ``ss ^= other``

        Keep only values found in exactly one of itself and `other`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.symmetric_difference_update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3, 6, 7])

        Args:
            other (Iterable[T]): `other` iterable

        Returns:
            Self: updated sorted set

        """
        set_ = self._set
        list_ = self._list
        set_.symmetric_difference_update(other)
        list_.clear()
        list_.update(set_)
        return self

    @override
    def __ixor__(self, other: Iterable[T]) -> Self:
        return self.symmetric_difference_update(other)

    @override
    def union(self, *iterables: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return new sorted set with values from itself and all `iterables`.

        The `union` method also corresponds to operator ``|``.

        ``ss.__or__(iterable)`` <==> ``ss | iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.union([4, 5, 6, 7])
        SortedSet([1, 2, 3, 4, 5, 6, 7])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        return self.__class__(self.iter().chain(*iterables))

    @override
    def __or__(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.union(other)

    def __ror__(self, other: Iterable[T]) -> Self:
        return self.union(other)

    def update(self, *iterables: Iterable[T]) -> Self:
        """Update the sorted set adding values from all `iterables`.

        The `update` method also corresponds to operator ``|=``.

        ``ss.__ior__(iterable)`` <==> ``ss |= iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3, 4, 5, 6, 7])

        Args:
            *iterables (Iterable[T]): iterable arguments

        Returns:
            Self: updated sorted set

        """
        list_ = self._list
        values = Iter(iterables).flatten().collect(Set)
        if (4 * values.len()) > len(self._set):
            list_ = self._list
            self._set.update(values)
            list_.clear()
            list_.update(self._set)
        else:
            for value in values:
                self.add(value)
        return self

    @override
    def __ior__(self, other: Iterable[T]) -> Self:
        return self.update(other)

    @override
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[AbstractSet[T]]]:
        """Support for pickle.

        The tricks played with exposing methods in :func:`SortedSet.__init__`
        confuse pickle so customize the reducer.

        Returns:
            tuple[type[Self], tuple[AbstractSet[T]]]: tuple of class and arguments

        """
        return (self.__class__, (self._set,))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted set.

        ``ss.__repr__()`` <==> ``repr(ss)``

        :return: string representation

        """
        type_name = self.__class__.__name__
        return f"{type_name}({list(self)!r})"


class SortedKeySet[T, OT: SupportsHashableAndRichComparison](SortedSet[T]):  # pyright: ignore[reportInvalidTypeArguments]
    _set: set[T]
    _list: SortedKeyList[T, OT]

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        key: SetKeyFunc[T, OT] = identity,  # pyright: ignore[reportArgumentType]
    ) -> None:
        self._key: SetKeyFunc[T, OT] = key
        """Initialize sorted set instance based on a key function.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted key set.

        The `key` argument defines a callable that, like the `key`
        argument to Python's `sorted` function, extracts a comparison key from
        each value. The default, none, compares values directly.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> ss = SortedKeySet([3, 1, 2, 5, 4], neg)
        >>> ss
        SortedKeySet([5, 4, 3, 2, 1], key=<built-in function neg>)

        :param iterable: initial values (optional)
        :param key: function used to extract comparison key

        """
        # SortedSet._fromset calls SortedSet.__init__ after initializing the
        # _set attribute. So only create a new set if the _set attribute is not
        # already present.

        if not hasattr(self, "_set"):
            self._set = set()

        self._list = SortedKeyList(self._set, key=key)  # pyright: ignore[reportIncompatibleVariableOverride]

        if iterable is not None:
            _ = self.update(iterable)

    @override
    def _fromset(self, values: set[T]) -> Self:
        sorted_set = self.__new__(self.__class__)
        sorted_set._set = values
        sorted_set.__init__(key=self._key)
        return sorted_set

    @property
    def key(self) -> SetKeyFunc[T, OT]:
        """Function used to extract comparison key from values.

        Sorted set compares values directly when the key function is none.

        """
        return self._key

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted set.

        ``ss.__repr__()`` <==> ``repr(ss)``

        :return: string representation

        """
        key_ = self._key
        key = f", key={key_!r}"
        type_name = self.__class__.__name__
        return f"{type_name}({list(self)!r}{key})"

    @override
    def __reduce__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> tuple[type[Self], tuple[AbstractSet[T], Callable[[T], Any]]]:
        return (self.__class__, (self._set, self._key))

    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        return self._list.irange_key(min_key, max_key, inclusive, reverse=reverse)

    def bisect_key_left(self, key: OT) -> int:
        return self._list.bisect_key_left(key)

    def bisect_key_right(self, key: OT) -> int:
        return self._list.bisect_key_right(key)

    @override
    def union(self, *iterables: Iterable[T]) -> Self:
        return self.__class__(self.iter().chain(*iterables), key=self._key)

    @override
    def symmetric_difference(self, other: Iterable[T]) -> Self:
        diff = self._set.symmetric_difference(other)
        return self._fromset(diff)

    @override
    def intersection(self, *iterables: Iterable[Any]) -> Self:
        intersect = self._set.intersection(*iterables)
        return self._fromset(intersect)

    @override
    def difference(self, *iterables: Iterable[Any]) -> Self:
        diff = self._set.difference(*iterables)
        return self._fromset(diff)

    @override
    def copy(self) -> Self:
        return self._fromset(set(self._set))
