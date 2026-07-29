from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final, Self, overload, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import NotImplementedType

    from _typeshed import SupportsRichComparison

    from pyochain import Vec
    from pyochain.abc import PyoIterator

    from ..rs import InnerSorted

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]
DEFAULT_LOAD_FACTOR: Final[int] = 1000


class SortedCollection[T](ABC):
    """Base class for sorted collections."""

    @abstractmethod
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Any, ...]]: ...
    @abstractmethod
    @override
    def __repr__(self) -> str: ...

    @abstractmethod
    def __contains__(self, value: object) -> bool: ...

    @abstractmethod
    def bisect_left(self, value: T) -> int:
        """Return an index to insert *value* in the `SortedCollection`.

        If the *value* is already present, the insertion point will be before
        (to the left of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): value to insert in `SortedCollection`.

        Returns:
            int: insertion index of value in `SortedCollection`.

        Examples:
            >>> from pyochain.collections import SortedList, SortedKeyList
            >>> sl = SortedList([10, 11, 12, 13, 14])
            >>> sl.bisect_left(12)
            2

            >>> from operator import neg
            >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
            >>> skl.bisect_left(1)
            4

        """

    @abstractmethod
    def bisect_right(self, value: T) -> int:
        """Return an index to insert `value` in the `SortedCollection`.

        Similar to `bisect_left`, but if `value` is already present, the
        insertion point will be after (to the right of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): value to insert in `SortedCollection`.

        Returns:
            int: insertion index of value in `SortedCollection`.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList([10, 11, 12, 13, 14])
        >>> sl.bisect_right(12)
        3

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_right(1)
        5

        """

    @abstractmethod
    def index(self, value: T, start: int | None = None, stop: int | None = None) -> int:
        """Return first index of value in sorted list.

        Raise ValueError if `value` is not present.

        Index must be between `start` and `stop` for the `value` to be
        considered present. The default value, None, for `start` and `stop`
        indicate the beginning and end of the sorted list.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList("abcde")
        >>> sl.index("d")
        3
        >>> sl.index("z")
        Traceback (most recent call last):
          ...
        ValueError: 'z' is not in list
        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.index(2)
        3
        >>> skl.index(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 is not in list

        Args:
            value (T): value in sorted list
            start (int | None): start index (default None, start of sorted list)
            stop (int | None): stop index (default None, end of sorted list)

        Raises:
            ValueError: if value is not present
        Returns:
            int: index of value

        """

    @abstractmethod
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        """Create an iterator of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically
        inclusive of the beginning and end of the `SortedCollection`.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.


        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList("abcdefghij")
        >>> it = sl.irange("c", "f")
        >>> list(it)
        ['c', 'd', 'e', 'f']
        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange(14.5, 11.5)
        >>> list(it)
        [14, 13, 12]

        Args:
            minimum (T | None): minimum value to start iterating
            maximum (T | None): maximum value to stop iterating
            inclusive (tuple[bool, bool]): pair of booleans
            reverse (bool): yield values in reverse order

        Returns:
            Iterator[T]: an iterator of values between `minimum` and `maximum`

        """

    @abstractmethod
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        """Return an iterator that slices sorted list from `start` to `stop`.

        The `start` and `stop` index are treated inclusive and exclusive,
        respectively.

        Both `start` and `stop` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from pyochain.collections import SortedList
        >>> sl = SortedList("abcdefghij")
        >>> it = sl.islice(2, 6)
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param int start: start index (inclusive)
        :param int stop: stop index (exclusive)
        :param bool reverse: yield values in reverse order
        :return: iterator

        """

    @abstractmethod
    def reset(self, load: int) -> None:
        """Reset sorted list load factor.

        The `load` specifies the load-factor of the list. The default load
        factor of 1000 works well for lists from tens to tens-of-millions of
        values.

        Good practice is to use a value that is the cube root of the list size.

        With billions of elements, the best load factor depends on your usage.

        It's best to leave the load factor at the default until you start benchmarking.

        Runtime complexity: `O(n)`

        :param int load: load-factor for sorted list sublists

        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all values from the `SortedCollection`.

        Runtime complexity: `O(n)`

        """


class BaseSortedListSet[T](SortedCollection[T], ABC):
    @abstractmethod
    def add(self, value: T) -> None:
        """Add `value` to the `SortedCollection`.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList()
        >>> sl.add(3)
        >>> sl.add(1)
        >>> sl.add(2)
        >>> sl
        SortedList([1, 2, 3])
        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.add(3)
        >>> skl.add(1)
        >>> skl.add(2)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        Args:
            value (T): value to add to the `SortedCollection`

        """

    @abstractmethod
    def discard(self, value: T) -> None:
        """Remove `value` from sorted-key list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.discard(5)
        >>> sl.discard(0)
        >>> sl == [1, 2, 3, 4]
        True

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.discard(1)
        >>> skl.discard(0)
        >>> skl == [5, 4, 3, 2]
        True

        :param value: `value` to discard from sorted-key list

        """

    @abstractmethod
    def remove(self, value: T, /) -> None:
        """Remove `value` from the `SortedCollection`.

        `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.remove(5)
        >>> sl == [1, 2, 3, 4]
        True
        >>> sl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list
        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> skl.remove(5)
        >>> skl == [4, 3, 2, 1]
        True
        >>> skl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list

        Args:
            value (T): `value` to remove from the `SortedCollection`

        Raises:
            ValueError: if `value` is not in the `SortedCollection`


        """

    @abstractmethod
    def copy(self) -> Self:
        """Return a shallow copy of the `SortedCollection`.

        Runtime complexity: `O(n)`

        :return: new sorted-key list

        """


class BaseSortedList[T](BaseSortedListSet[T], ABC):  # ruff:ignore[eq-without-hash]
    _inner: InnerSorted[T, Any]

    def __iter__(self) -> PyoIterator[T]:
        """Return an iterator over the sorted list.

        ``sl.__iter__()`` <==> ``iter(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return self._inner.iter()

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted list.

        ``sl.__contains__(value)`` <==> ``value in sl``

        Runtime complexity: `O(log(n))`

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> 3 in sl
        True
        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> 3 in skl
        True

        Args:
            value (object): search for value in sorted list
        Returns:
            (bool): `True` if `value` in sorted list.


        """
        return self._inner.contains(value)

    @abstractmethod
    def __add__(self, other: Iterable[T]) -> Self: ...

    @abstractmethod
    def __mul__(self, num: int) -> Self: ...

    @override
    def __eq__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is equal to `other`.

        ``sl.__eq__(other)`` <==> ``sl == other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is equal to `other`

        """
        return self._inner.eq(other)

    @override
    def __ne__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is not equal to `other`.

        ``sl.__ne__(other)`` <==> ``sl != other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is not equal to `other`

        """
        return self._inner.ne(other)

    def __lt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than `other`.

        ``sl.__lt__(other)`` <==> ``sl < other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is less than `other`

        """
        return self._inner.lt(other)

    def __gt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than `other`.

        ``sl.__gt__(other)`` <==> ``sl > other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is greater than `other`

        """
        return self._inner.gt(other)

    def __le__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than or equal to `other`.

        ``sl.__le__(other)`` <==> ``sl <= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is less than or equal to `other`

        """
        return self._inner.le(other)

    def __ge__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than or equal to `other`.

        ``sl.__ge__(other)`` <==> ``sl >= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is greater than or equal to `other`

        """
        return self._inner.ge(other)

    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted list.

        ``sl.__delitem__(index)`` <==> ``del sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList
        >>> sl = SortedList("abcde")
        >>> del sl[2]
        >>> sl
        SortedList(['a', 'b', 'd', 'e'])
        >>> del sl[:2]
        >>> sl
        SortedList(['d', 'e'])

        :param index: integer or slice for indexing
        :raises IndexError: if index out of range

        """
        return self._inner.delitem(index)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[T]: ...
    def __getitem__(self, index: int | slice) -> T | Vec[T]:
        """Lookup value at `index` in sorted list.

        ``sl.__getitem__(index)`` <==> ``sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList
        >>> sl = SortedList("abcde")
        >>> sl[1]
        'b'
        >>> sl[-1]
        'e'
        >>> sl[2:5]
        Vec('c', 'd', 'e')

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            T | Vec[T]: value or list of values
        """
        return self._inner.getitem(index)

    @override
    def add(self, value: T) -> None:
        return self._inner.add(value)

    @override
    def discard(self, value: T) -> None:
        return self._inner.discard(value)

    @override
    def remove(self, value: T, /) -> None:
        return self._inner.remove(value)

    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        >>> sl.count(3)
        3
        >>> from operator import neg
        >>> skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)
        >>> skl.count(2)
        2

        Args:
            value (T): value to count in sorted list
        Returns:
            count
        """
        return self._inner.count(value)

    def _loc(self, pos: int, idx: int) -> int:
        return self._inner.loc(pos, idx)

    def _build_index(self) -> None:
        return self._inner.build_index()

    def _pos(self, idx: int) -> tuple[int, int]:
        return self._inner.pos(idx)

    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted list.

        Raise :exc:`IndexError` if the sorted list is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList("abcde")
        >>> sl.pop()
        'e'
        >>> sl.pop(2)
        'c'
        >>> sl
        SortedList(['a', 'b', 'd'])

        Args:
            index (int): index of value (default -1)

        Returns:
            T: value at `index` in sorted list
        """
        return self._inner.pop(index)

    @override
    def bisect_left(self, value: T) -> int:
        return self._inner.bisect_left(value)

    @override
    def bisect_right(self, value: T) -> int:
        return self._inner.bisect_right(value)

    @override
    def index(self, value: T, start: int | None = None, stop: int | None = None) -> int:
        return self._inner.index(value, start, stop)

    @override
    def reset(self, load: int) -> None:
        return self._inner.reset(load)

    def update(self, iterable: Iterable[T]) -> None:
        """Add all the values from *iterable* to the `SortedCollection`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> from pyochain.collections import SortedList, SortedKeyList
        >>> sl = SortedList()
        >>> sl.update([3, 1, 2])
        >>> sl
        SortedList([1, 2, 3])

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.update([3, 1, 2])
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param iterable: iterable of values to add

        """
        return self._inner.update(iterable)
