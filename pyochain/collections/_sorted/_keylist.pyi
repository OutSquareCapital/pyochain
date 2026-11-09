# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Self, final, override

from _typeshed import SupportsRichComparison

from ._core import KeyedCollection, KeyFunc
from ._list import BaseSortedList

@final
class SortedKeyList[T, OT: SupportsRichComparison](
    BaseSortedList[T], KeyedCollection[T, OT]
):
    """a `MutableSequence` that maintains its values in sorted order based on a key function.

    The sorted-key list maintains values in comparison order based on the result of a key function applied to every value.

    Optional `iterable` argument provides an initial iterable of values to initialize the sorted-key list.

    `key` argument defines a `Callable` that, like the `key` argument to Python's `sorted` function, extracts a comparison key from each value.

    Runtime complexity: `O(n*log(n))`

    ```python
    from pyochain.collections import SortedKeyList
    from operator import neg

    skl = SortedKeyList(neg)
    assert repr(skl) == "SortedKeyList(<built-in function neg>, [])"
    skl = SortedKeyList(neg, [3, 1, 2])
    assert repr(skl) == "SortedKeyList(<built-in function neg>, [3, 2, 1])"
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
    def copy(self) -> Self: ...
