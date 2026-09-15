# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Self, final, override

from pyochain._types import SupportsHashableAndRichComparison
from pyochain.abc import PyoIterator

from ._core import KeyFunc
from ._set import BaseSortedSet

type SetKeyFunc[T, OT: SupportsHashableAndRichComparison] = KeyFunc[T, OT]

@final
# pyrefly: ignore [bad-specialization]
class SortedKeySet[T, OT: SupportsHashableAndRichComparison](BaseSortedSet[T]):  # pyright: ignore[reportInvalidTypeArguments] # ty: ignore[invalid-type-arguments]
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
            assert (
                repr(ss) == "SortedKeySet([5, 4, 3, 2, 1], key=<built-in function neg>)"
            )
            ```

        """

    def irange_key[T1, OT1: SupportsHashableAndRichComparison](
        self: SortedKeySet[T1, OT1],
        min_key: OT1 | None = None,
        max_key: OT1 | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T1]: ...
    def bisect_key_left(self, key: OT) -> int: ...
    def bisect_key_right(self, key: OT) -> int: ...
    @override
    def union[T1, OT1: SupportsHashableAndRichComparison](
        self: SortedKeySet[T1, OT1], *iterables: Iterable[T1]
    ) -> SortedKeySet[T1, OT]: ...
