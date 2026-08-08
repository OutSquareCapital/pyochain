# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from reprlib import recursive_repr
from typing import TYPE_CHECKING, Self, override

from pyochain.rs import InnerLists

from ._base_sorted import BaseSortedList, SortedCollection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import SupportsRichComparison

    from pyochain import Vec


class SortedList[T: SupportsRichComparison](BaseSortedList[T], SortedCollection[T]):
    """Sorted list is a sorted mutable sequence.

    Sorted list values are maintained in sorted order.

    Sorted list values must be comparable. The total ordering of values must
    not change while they are stored in the sorted list.

    Methods for adding values:

    * :func:`SortedList.add`
    * :func:`SortedList.update`
    * :func:`SortedList.__add__`
    * :func:`SortedList.__iadd__`
    * :func:`SortedList.__mul__`
    * :func:`SortedList.__imul__`

    Methods for removing values:

    * :func:`SortedList.clear`
    * :func:`SortedList.discard`
    * :func:`SortedList.remove`
    * :func:`SortedList.pop`
    * :func:`SortedList.__delitem__`

    Methods for looking up values:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.__contains__`
    * :func:`SortedList.__getitem__`

    Methods for iterating values:

    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.__iter__`
    * :func:`SortedList.__reversed__`

    Methods for miscellany:

    * :func:`SortedList.copy`
    * :func:`SortedList.__len__`
    * :func:`SortedList.__repr__`
    * :func:`SortedList._check`
    * :func:`SortedList.reset`

    Sorted lists use lexicographical ordering semantics when compared to other
    sequences.

    Some methods of mutable sequences are not supported and will raise
    not-implemented error.

    Optional `iterable` argument provides an initial iterable of values to
    initialize the sorted list.

    Runtime complexity: `O(n*log(n))`

    ```python
    from pyochain.collections import SortedList

    sl = SortedList()
    assert repr(sl) == "SortedList([])"
    sl = SortedList([3, 1, 2, 5, 4])
    assert repr(sl) == "SortedList([1, 2, 3, 4, 5])"
    ```

    """

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._inner: InnerLists[T, T] = InnerLists[T, T]()  # pyright: ignore[reportIncompatibleVariableOverride]

        if iterable is not None:
            self._inner.update(iterable)

    @property
    def inner(self) -> InnerLists[T, T]:
        return self._inner

    @override
    def clear(self) -> None:
        self._inner.clear()

    @override
    def copy(self) -> Self:
        return self.__class__(self)

    def __copy__(self) -> Self:
        return self.copy()

    @override
    def __add__(self, other: Iterable[T]) -> Self:
        values = self._inner.collapse_lists()
        values.extend(other)
        return self.__class__(values)

    @override
    def __mul__(self, num: int) -> Self:
        values = self._inner.collapse_lists().repeat(num)
        return self.__class__(values)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T]]]:
        values = self._inner.collapse_lists()
        return (self.__class__, (values,))

    @recursive_repr()
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)!r})"
