# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from abc import ABC
from collections.abc import Callable
from typing import Self, type_check_only

from _typeshed import SupportsRichComparison

from pyochain.abc import DoubleEndedIterator

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]

@type_check_only
class SortedCollection[T](ABC):
    """Base class for sorted collections."""
    def __contains__(self, value: object, /) -> bool: ...
    def bisect_left(self, value: T) -> int:
        """Return an index to insert *value* in the `SortedCollection`.

        If the *value* is already present, the insertion point will be before (to the left of) any existing values.

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

        skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
        assert skl.bisect_left(1) == 4
        ```
        """

    def bisect_right(self, value: T) -> int:
        """Return an index to insert `value` in the `SortedCollection`.

        Similar to [`bisect_left`][] but if *value* is already present, the
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

            skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
            assert skl.bisect_right(1) == 5
            ```
        """
    def index(
        self, value: T, start: int | None = None, stop: int | None = None, /
    ) -> int:
        """Return first index of value in the `SortedCollection`.

        Raise `ValueError` if *value* is not present.

        Index must be between *start* and *stop* for the *value* to be considered present.

        The default value, `None`, for *start* and *stop* indicate the beginning and end of the `SortedCollection`.

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
            skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
            assert skl.index(2) == 3
            with pytest.raises(ValueError):
                skl.index(0)
            ```
        """

    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
    ) -> DoubleEndedIterator[T]:
        """Create a [`DoubleEndedIterator`][] of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically inclusive of the beginning and end of the `SortedCollection`.

        Args:
            minimum (T | None): minimum value to start iterating
            maximum (T | None): maximum value to stop iterating
            inclusive (tuple[bool, bool]): Whether the minimum and maximum ought to be included in the range, respectively. The default is `(True, True)` such that the range is inclusive of both minimum and maximum.

        Returns:
            DoubleEndedIterator[T]: an `Iterator` of values between `minimum` and `maximum`

        Examples:
            ```python
            from pyochain.collections import SortedList, SortedKeyList
            from operator import neg

            sl = SortedList("abcdefghij")
            it = sl.irange("c", "f")
            assert list(it) == ["c", "d", "e", "f"]

            skl = SortedKeyList(neg, [11, 12, 13, 14, 15])
            it = skl.irange(14.5, 11.5)
            assert list(it) == [14, 13, 12]
            ```
        """
    def islice(
        self, start: int | None = None, stop: int | None = None
    ) -> DoubleEndedIterator[T]:
        """Return a [`DoubleEndedIterator`][] that slices the `SortedCollection` from `start` to `stop`.

        The `start` and `stop` index are treated inclusive and exclusive, respectively.

        Both `start` and `stop` default to `None` which is automatically inclusive of the beginning and end of the sorted list.

        Args:
            start (int | None): start index (inclusive)
            stop (int | None): stop index (exclusive)

        Returns:
            DoubleEndedIterator[T]: an `Iterator` over a slice from `start` to `stop`

        Examples:
            ```python
            from pyochain.collections import SortedList

            sl = SortedList("abcdefghij")
            it = sl.islice(2, 6)
            assert list(it) == ["c", "d", "e", "f"]
            ```
        """
    def reset(self, load: int) -> None:
        """Reset the load factor of the `SortedCollection`.

        The `load` specifies the load-factor of the list.

        The default load factor of 1000 works well for collections from tens to tens-of-millions of values.

        Good practice is to use a value that is the cube root of the list size.

        With billions of elements, the best load factor depends on your usage.

        It's best to leave the load factor at the default until you start benchmarking.

        Runtime complexity: `O(n)`

        Args:
            load (int): load-factor for the sub-vectors of the `SortedCollection`

        """
    def clear(self) -> None:
        """Remove all values from the `SortedCollection`.

        Runtime complexity: `O(n)`

        """

class BaseSortedListSet[T](SortedCollection[T], ABC):
    def discard(self, value: T) -> None:
        """Remove `value` from the `SortedCollection` if it is a member.

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

        skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
        skl.discard(1)
        skl.discard(0)
        assert skl == [5, 4, 3, 2]
        ```
        """
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
        skl = SortedKeyList(neg, [1, 2, 3, 4, 5])
        skl.remove(5)
        assert skl == [4, 3, 2, 1]

        try:
            skl.remove(0)
        except ValueError as e:
            assert str(e) == "0 not in list"
        ```
        """
    def copy(self) -> Self:
        """Return a shallow copy of the `SortedCollection`.

        Runtime complexity: `O(n)`

        Returns:
            Self: new sorted-key list

        """

class KeyedCollection[T, OT: SupportsRichComparison](SortedCollection[T], ABC):
    """Common interface for sorted collections whose values are sorted based on a key function."""
    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
    ) -> DoubleEndedIterator[T]:
        """Create a [`DoubleEndedIterator`][] of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically inclusive of the beginning and end of `Self`.

        Args:
            min_key (OT | None): minimum key to start iterating
            max_key (OT | None): maximum key to stop iterating
            inclusive (tuple[bool, bool]): Whether the minimum and maximum ought to be included in the range, respectively. The default is ``(True, True)`` such that the range is inclusive of both minimum and maximum.

        Returns:
            DoubleEndedIterator[T]: `Iterator` of values between `min_key` and `max_key`

        Examples:
        ```python
        from pyochain.collections import SortedKeyList
        from operator import neg

        skl = SortedKeyList(neg, [11, 12, 13, 14, 15])
        it = skl.irange_key(-14, -12)
        assert list(it) == [14, 13, 12]
        ```

        """

    def bisect_key_left(self, key: OT) -> int:
        """Return an index to insert `key` in the `SortedCollection`.

        If the `key` is already present, the insertion point will be before (to the left of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            key (OT): insertion index of key in sorted-key list

        Returns:
            int: index

        Examples:
        ```python
        from pyochain.collections import SortedKeyList
        from operator import neg

        skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
        assert skl.bisect_key_left(-1) == 4
        ```
        """

    def bisect_key_right(self, key: OT) -> int:
        """Return an index to insert `key` in the `SortedCollection`.

        Similar to `bisect_key_left`, but if `key` is already present, the insertion point will be after (to the right of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            key (OT): insertion index of key in sorted-key list

        Returns:
            int: index

        Examples:
        ```python
        from pyochain.collections import SortedKeyList
        from operator import neg

        skl = SortedKeyList(neg, [5, 4, 3, 2, 1])
        assert skl.bisect_key_right(-1) == 5
        ```
        """
