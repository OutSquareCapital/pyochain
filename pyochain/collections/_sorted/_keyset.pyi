# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from typing import Any, Self, final, override

from pyochain.abc import PyoIterator

from ._core import KeyFunc
from ._set import BaseSortedSet
from ._views import SupportsHashableAndRichComparison

type SetKeyFunc[T, OT: SupportsHashableAndRichComparison] = KeyFunc[T, OT]

@final
# pyrefly: ignore [bad-specialization]
class SortedKeySet[T, OT: SupportsHashableAndRichComparison](BaseSortedSet[T]):  # pyright: ignore[reportInvalidTypeArguments]
    def __new__(
        cls, iterable: Iterable[T] | None = None, key: SetKeyFunc[T, OT] | None = None
    ) -> Self:
        """Initialize sorted set instance based on a key function.

        Optional `iterable` argument provides an initial iterable of values to initialize the sorted key set.

        The `key` argument defines a `Callable` that, like the `key` argument to Python's `sorted` function, extracts a comparison key from each value.

        The default, `None`, compares values directly.

        Runtime complexity: `O(n*log(n))`

        Args:
            iterable (Iterable[T] | None): initial values (optional)
            key (SetKeyFunc[T, OT] | None): function used to extract comparison key

        Returns:
            Self: new sorted-key set

        Examples:
            ```python
            from pyochain.collections import SortedKeySet
            from operator import neg

            ss = SortedKeySet([3, 1, 2, 5, 4], neg)
            assert (
                repr(ss) == "SortedKeySet([5, 4, 3, 2, 1], key=<built-in function neg>)"
            )
            ```

        """

    @override
    # pyrefly: ignore [bad-override]
    def __reduce__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> tuple[type[Self], tuple[AbstractSet[T], Callable[[T], Any]]]: ...
    @property
    def key(self) -> SetKeyFunc[T, OT]:
        """Function used to extract comparison key from values.

        Sorted set compares values directly when the key function is none.

        """

    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
    def bisect_key_left(self, key: OT) -> int: ...
    def bisect_key_right(self, key: OT) -> int: ...
    @override
    def union(self, *iterables: Iterable[T]) -> Self: ...
