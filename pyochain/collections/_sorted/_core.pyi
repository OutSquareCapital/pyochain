# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self, override

from _typeshed import SupportsRichComparison

from pyochain.abc import PyoIterator

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]

class SortedCollection[T](ABC):
    """Base class for sorted collections."""

    @abstractmethod
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Any, ...]]: ...
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
            import pytest

            sl = SortedList("abcde")
            assert sl.index("d") == 3
            with pytest.raises(ValueError):
                sl.index("z")
            skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
            assert skl.index(2) == 3
            with pytest.raises(ValueError):
                skl.index(0)
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
            Self: new sorted-key list

        """
