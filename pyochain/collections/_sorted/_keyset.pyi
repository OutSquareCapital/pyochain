# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Self, final, override

from pyochain._types import SupportsHashableAndRichComparison

from ._core import KeyedCollection, KeyFunc
from ._set import BaseSortedSet

type SetKeyFunc[T, OT: SupportsHashableAndRichComparison] = KeyFunc[T, OT]

@final
# pyrefly: ignore [bad-specialization]
class SortedKeySet[T, OT: SupportsHashableAndRichComparison](
    BaseSortedSet[T],  # pyright: ignore[reportInvalidTypeArguments] # ty: ignore[invalid-type-arguments],
    KeyedCollection[T, OT],
):
    def __new__(
        cls, key: SetKeyFunc[T, OT], iterable: Iterable[T] | None = None, /
    ) -> Self:
        """Initialize sorted set instance based on a key function.

        Optional *iterable* argument provides an initial iterable of values to initialize the sorted key set.

        The `key` argument defines a `Callable` that, like the `key` argument to Python's `sorted` function, extracts a comparison key from each value.

        Runtime complexity: `O(n*log(n))`

        Args:
            key (SetKeyFunc[T, OT]): function used to extract comparison key
            iterable (Iterable[T] | None): initial values (optional)

        Returns:
            Self: new sorted-key set

        Examples:
            ```python
            from pyochain.collections import SortedKeySet
            from operator import neg

            ss = SortedKeySet(neg, [3, 1, 2, 5, 4])
            assert repr(ss) == "SortedKeySet(<built-in function neg>, [5, 4, 3, 2, 1])"
            ```

        """
    @override
    def union[T1, OT1: SupportsHashableAndRichComparison](
        self: SortedKeySet[T1, OT1], *iterables: Iterable[T1]
    ) -> SortedKeySet[T1, OT]: ...
