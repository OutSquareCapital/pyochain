# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Self, override

from _typeshed import SupportsRichComparison

from pyochain import Vec
from pyochain.abc import PyoIterator

from ._core import KeyFunc
from ._list import BaseSortedList

class SortedKeyList[T, OT: SupportsRichComparison](BaseSortedList[T]):
    """a `MutableSequence` that maintains its values in sorted order based on a key function.

    The sorted-key list maintains values in comparison order based on the result of a key function applied to every value.

    Optional `iterable` argument provides an initial iterable of values to initialize the sorted-key list.

    `key` argument defines a `Callable` that, like the `key` argument to Python's `sorted` function, extracts a comparison key from each value.

    Runtime complexity: `O(n*log(n))`

    ```python
    from pyochain.collections import SortedKeyList
    from operator import neg

    skl = SortedKeyList(neg)
    assert repr(skl) == "SortedKeyList([], key=<built-in function neg>)"
    skl = SortedKeyList(neg, [3, 1, 2])
    assert repr(skl) == "SortedKeyList([3, 2, 1], key=<built-in function neg>)"
    ```
    """
    def __new__(
        cls, key: KeyFunc[T, OT], iterable: Iterable[T] | None = None, /
    ) -> Self:
        """Create a new sorted-key list.

        Args:
            key (KeyFunc[T, OT]): function used to extract comparison key.
            iterable (Iterable[T] | None): initial values (optional)

        Returns:
            Self: new sorted-key list
        """

    @override
    def __add__(self, other: Iterable[T]) -> Self: ...
    @override
    def __mul__(self, num: int) -> Self: ...
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T], KeyFunc[T, OT]]]: ...
    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        """Create an iterator of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically inclusive of the beginning and end of the sorted-key list.

        Args:
            min_key (OT | None): minimum key to start iterating
            max_key (OT | None): maximum key to stop iterating
            inclusive (tuple[bool, bool]): Whether the minimum and maximum ought to be included in the range, respectively. The default is ``(True, True)`` such that the range is inclusive of both minimum and maximum.
            reverse (bool): When `True` the values are yielded in reverse order. Defaults to `False`.

        Returns:
            PyoIterator[T]: iterator of values between `min_key` and `max_key`

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
        """Return an index to insert `key` in the `SortedKeyList`.

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
        """Return an index to insert `key` in the `SortedKeyList`.

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

    @override
    def copy(self) -> Self: ...
