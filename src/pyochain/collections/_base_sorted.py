from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final, Self, overload, override

from pyochain import Iter

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
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg


        sl = SortedList([10, 11, 12, 13, 14])
        assert sl.bisect_left(12) == 2

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        assert skl.bisect_left(1) == 4
        ```
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

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([10, 11, 12, 13, 14])
        assert sl.bisect_right(12) == 3

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        assert skl.bisect_right(1) == 5
        ```
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

        Args:
            value (T): value in sorted list
            start (int | None): start index (default None, start of sorted list)
            stop (int | None): stop index (default None, end of sorted list)


        Returns:
            int: index of value

        Raises:
            ValueError: if value is not present

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList("abcde")
        assert sl.index("d") == 3
        try:
            sl.index("z")
        except ValueError as e:
            assert str(e) == "'z' is not in list"

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        assert skl.index(2) == 3
        try:
            skl.index(0)
        except ValueError as e:
            assert str(e) == "0 is not in list"
        ```
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


        Args:
            minimum (T | None): minimum value to start iterating
            maximum (T | None): maximum value to stop iterating
            inclusive (tuple[bool, bool]): pair of booleans
            reverse (bool): yield values in reverse order

        Returns:
            PyoIterator[T]: an iterator of values between `minimum` and `maximum`

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList("abcdefghij")
        it = sl.irange("c", "f")
        assert list(it) == ["c", "d", "e", "f"]

        skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        it = skl.irange(14.5, 11.5)
        assert list(it) == [14, 13, 12]
        ```
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

        Args:
            start (int | None): start index (inclusive)
            stop (int | None): stop index (exclusive)
            reverse (bool): yield values in reverse order

        Returns:
            PyoIterator[T]: iterator

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl = SortedList("abcdefghij")
        it = sl.islice(2, 6)
        assert list(it) == ["c", "d", "e", "f"]
        ```
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

        Args:
            load (int): load-factor for sorted list sublists

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

        Args:
            value (T): value to add to the `SortedCollection`

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList()
        sl.add(3)
        sl.add(1)
        sl.add(2)
        assert sl == [1, 2, 3]

        skl = SortedKeyList(key=neg)
        skl.add(3)
        skl.add(1)
        skl.add(2)
        assert skl == [3, 2, 1]
        ```
        """

    @abstractmethod
    def discard(self, value: T) -> None:
        """Remove `value` from sorted-key list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): `value` to discard from sorted-key list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 3, 4, 5])
        sl.discard(5)
        sl.discard(0)
        assert sl == [1, 2, 3, 4]

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        skl.discard(1)
        skl.discard(0)
        assert skl == [5, 4, 3, 2]
        ```
        """

    @abstractmethod
    def remove(self, value: T, /) -> None:
        """Remove `value` from the `SortedCollection`.

        `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): `value` to remove from the `SortedCollection`

        Raises:
            ValueError: if `value` is not in the `SortedCollection`

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 3, 4, 5])
        sl.remove(5)
        assert sl == [1, 2, 3, 4]
        try:
            sl.remove(0)
        except ValueError as e:
            assert str(e) == "0 not in list"
        skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        skl.remove(5)
        assert skl == [4, 3, 2, 1]

        try:
            skl.remove(0)
        except ValueError as e:
            assert str(e) == "0 not in list"
        ```
        """

    @abstractmethod
    def copy(self) -> Self:
        """Return a shallow copy of the `SortedCollection`.

        Runtime complexity: `O(n)`

        Returns:
            (Self): new sorted-key list

        """


class BaseSortedList[T](BaseSortedListSet[T], ABC):  # ruff:ignore[eq-without-hash]
    _inner: InnerSorted[T, Any]

    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        match self._inner.islice_specs(start, stop):
            case None:
                return Iter(())
            case (min_pos, min_idx, max_pos, max_idx):
                return self._islice(min_pos, min_idx, max_pos, max_idx, reverse=reverse)

    def _islice(
        self, min_pos: int, min_idx: int, max_pos: int, max_idx: int, *, reverse: bool
    ) -> PyoIterator[T]:
        """Return an iterator that slices sorted list using two index pairs.

        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the
        first inclusive and the latter exclusive. See `_pos` for details on how
        an index is converted to an index pair.

        When `reverse` is `True`, values are yielded from the iterator in
        reverse order.

        """
        return self._inner.islice_iter(
            min_pos, min_idx, max_pos, max_idx, reverse=reverse
        )

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

        Args:
            value (object): search for value in sorted list

        Returns:
            bool: `True` if `value` in sorted list.

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 3, 4, 5])
        assert 3 in sl

        skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        assert 3 in skl
        ```

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

        Args:
            other (object): `other` sequence

        Returns:
            (NotImplementedType | bool): true if sorted list is equal to `other`

        """
        return self._inner.eq(other)

    @override
    def __ne__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is not equal to `other`.

        ``sl.__ne__(other)`` <==> ``sl != other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            (NotImplementedType | bool): true if sorted list is not equal to `other`

        """
        return self._inner.ne(other)

    def __lt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than `other`.

        ``sl.__lt__(other)`` <==> ``sl < other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence
        Returns:
            (NotImplementedType | bool): true if sorted list is less than `other`

        """
        return self._inner.lt(other)

    def __gt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than `other`.

        ``sl.__gt__(other)`` <==> ``sl > other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            (NotImplementedType | bool): true if sorted list is greater than `other`

        """
        return self._inner.gt(other)

    def __le__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than or equal to `other`.

        ``sl.__le__(other)`` <==> ``sl <= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            (NotImplementedType | bool): true if sorted list is less than or equal to `other`

        """
        return self._inner.le(other)

    def __ge__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than or equal to `other`.

        ``sl.__ge__(other)`` <==> ``sl >= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            (NotImplementedType | bool): true if sorted list is greater than or equal to `other`

        """
        return self._inner.ge(other)

    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted list.

        ``sl.__delitem__(index)`` <==> ``del sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            index: integer or slice for indexing

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl = SortedList("abcde")

        del sl[2]
        assert sl == ["a", "b", "d", "e"]

        del sl[:2]
        assert sl == ["d", "e"]
        ```
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

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            T | Vec[T]: value or list of values

        Examples:
        ```python
        from pyochain.collections import SortedList
        from pyochain import Vec

        sl = SortedList("abcde")

        assert sl[1] == "b"
        assert sl[-1] == "e"
        assert sl[2:5] == Vec(("c", "d", "e"))
        ```
        """
        return self._inner.getitem(index)

    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return self._inner.reversed()

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

        Args:
            value (T): value to count in sorted list

        Returns:
            int: count of occurrences of `value` in sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        assert sl.count(3) == 3

        skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)
        assert skl.count(2) == 2
        ```
        """
        return self._inner.count(value)

    def _build_index(self) -> None:
        return self._inner.build_index()

    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted list.

        Raise :exc:`IndexError` if the sorted list is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            index (int): index of value (default -1)

        Returns:
            T: value at `index` in sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList

        sl = SortedList("abcde")
        assert sl.pop() == "e"
        assert sl.pop(2) == "c"
        assert sl == SortedList(["a", "b", "d"])
        ```
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

        Args:
            iterable (Iterable[T]): iterable of values to add

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList

        sl = SortedList()
        sl.update([3, 1, 2])
        assert sl == SortedList([1, 2, 3])

        from operator import neg

        skl = SortedKeyList(key=neg)
        skl.update([3, 1, 2])
        assert skl == [3, 2, 1]
        ```
        """
        return self._inner.update(iterable)
