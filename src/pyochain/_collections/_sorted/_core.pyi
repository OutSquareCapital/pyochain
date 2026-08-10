from abc import ABC, abstractmethod
from typing import Self

from pyochain.collections._base_sorted import (  # ruff: ignore[import-private-name]
    SortedCollection,
)

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
