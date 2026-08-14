# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Self, overload, override

from _typeshed import SupportsRichComparison

from pyochain import Vec
from pyochain.abc import PyoIterator

from ._core import KeyFunc
from ._list import BaseSortedList

class SortedKeyList[T, OT: SupportsRichComparison](BaseSortedList[T]):
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

    Optional `iterable` argument provides an initial iterable of values to
    initialize the sorted-key list.

    `key` argument defines a callable that, like the `key`
    argument to Python's `sorted` function, extracts a comparison key from
    each value. The default is the identity function.

    Runtime complexity: `O(n*log(n))`

    ```python
    from pyochain.collections import SortedKeyList
    from operator import neg

    skl = SortedKeyList(key=neg)
    assert repr(skl) == "SortedKeyList([], key=<built-in function neg>)"
    skl = SortedKeyList([3, 1, 2], key=neg)
    assert repr(skl) == "SortedKeyList([3, 2, 1], key=<built-in function neg>)"
    ```
    """
    @overload
    def __new__(
        cls, iterable: Iterable[OT], key: None = None
    ) -> SortedKeyList[OT, OT]: ...
    @overload
    def __new__(
        cls, iterable: Iterable[T] | None = None, key: KeyFunc[T, OT] = ...
    ) -> Self: ...
    def __new__(
        cls, iterable: Iterable[T] | None = None, key: KeyFunc[T, OT] | None = None
    ) -> Self:
        """Create a new sorted-key list.

        Args:
            iterable (Iterable[T] | None): initial values (optional)
            key (KeyFunc[T, OT] | None): function used to extract comparison key (optional)

        Returns:
            Self: new sorted-key list
        """
    @property
    def key(self) -> KeyFunc[T, OT]:
        """Function used to extract comparison key from values."""
    def irange_key(
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

        Args:
            min_key (OT | None): minimum key to start iterating
            max_key (OT | None): maximum key to stop iterating
            inclusive (tuple[bool, bool]): pair of booleans
            reverse (bool): yield values in reverse order

        Returns:
            PyoIterator[T]: iterator of values between `min_key` and `max_key`

        Examples:
        ```python
        from pyochain.collections import SortedKeyList
        from operator import neg

        skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        it = skl.irange_key(-14, -12)
        assert list(it) == [14, 13, 12]
        ```

        """
    def bisect_key_left(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        If the `key` is already present, the insertion point will be before (to
        the left of) any existing keys.

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

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        assert skl.bisect_key_left(-1) == 4
        ```
        """

    def bisect_key_right(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        Similar to `bisect_key_left`, but if `key` is already present, the
        insertion point will be after (to the right of) any existing keys.

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

        skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        assert skl.bisect_key_right(-1) == 5
        ```
        """

    @override
    def copy(self) -> Self: ...
    @override
    def __add__(self, other: Iterable[T]) -> Self: ...
    @override
    def __mul__(self, num: int) -> Self: ...
    @override
    # pyrefly: ignore [bad-override]
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T], KeyFunc[T, OT]]]: ...
