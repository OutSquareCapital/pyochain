# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from reprlib import recursive_repr
from typing import TYPE_CHECKING, Self, overload, override

from pyochain import Iter, Vec
from pyochain.abc import PyoIterator, PyoMutableSequence
from pyochain.rs import InnerLists

from ._base_sorted import BaseSortedList, SortedCollection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import SupportsRichComparison


class SortedList[T: SupportsRichComparison](
    BaseSortedList[T], SortedCollection[T], PyoMutableSequence[T]
):
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

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        """Raise not-implemented error.

        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``

        Raises:
            NotImplementedError: use ``del sl[index]`` and ``sl.add(value)`` instead

        """
        message = "use ``del sl[index]`` and ``sl.add(value)`` instead"
        raise NotImplementedError(message)

    @override
    def reverse(self) -> None:
        """Raise not-implemented error.

        Sorted list maintains values in ascending sort order. Values may not be
        reversed in-place.

        Use ``reversed(sl)`` for an iterator over values in descending sort
        order.

        Implemented to override `MutableSequence.reverse` which provides an
        erroneous default implementation.

        Raises:
            NotImplementedError: use ``reversed(sl)`` instead

        """
        msg = "use ``reversed(sl)`` instead"
        raise NotImplementedError(msg)

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]:
        match self._inner.irange(minimum, maximum, inclusive=inclusive):
            case None:
                return Iter(())
            case (min_pos, min_idx, max_pos, max_idx):
                return self._islice(min_pos, min_idx, max_pos, max_idx, reverse=reverse)

    @override
    def __len__(self) -> int:
        """Return the size of the sorted list.

        ``sl.__len__()`` <==> ``len(sl)``

        Returns:
            (int): size of sorted list

        """
        return self._inner.len

    @override
    def copy(self) -> Self:
        return self.__class__(self)

    def __copy__(self) -> Self:
        return self.copy()

    @override
    def append(self, value: T) -> None:
        """Raise not-implemented error.

        Implemented to override `MutableSequence.append` which provides an
        erroneous default implementation.

        Raises:
            NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def extend(self, values: object) -> None:
        """Raise not-implemented error.

        Implemented to override `MutableSequence.extend` which provides an
        erroneous default implementation.

        Raises:
            NotImplementedError: use ``sl.update(values)`` instead

        """
        msg = "use ``sl.update(values)`` instead"
        raise NotImplementedError(msg)

    @override
    def insert(self, index: int, value: T) -> None:
        """Raise not-implemented error.

        Raises:
            NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted list containing all values in both sequences.

        ``sl.__add__(other)`` <==> ``sl + other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(n*log(n))`

        Args:
            other (Iterable[T]): other iterable

        Returns:
            (Self): new sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl1 = SortedList("bat")
        sl2 = SortedList("cat")
        added = sl1 + sl2
        assert added == SortedList(["a", "a", "b", "c", "t", "t"])
        ```

        """
        values = self._inner.collapse_lists()
        values.extend(other)
        return self.__class__(values)

    def __radd__(self, other: Iterable[T]) -> Self:
        return self.__add__(other)

    @override
    def __iadd__(self, other: Iterable[T]) -> Self:
        """In-place update of the sorted list with values from `other`.

        ``sl.__iadd__(other)`` <==> ``sl += other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(k*log(n))` -- approximate.

        Args:
            other (Iterable[T]): other iterable

        Returns:
            Self: existing sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl = SortedList("bat")
        sl += "cat"
        assert sl == SortedList(["a", "a", "b", "c", "t", "t"])
        ```
        """
        self.update(other)
        return self

    @override
    def __mul__(self, num: int) -> Self:
        """Return new sorted list with `num` shallow copies of values.

        ``sl.__mul__(num)`` <==> ``sl * num``

        Runtime complexity: `O(n*log(n))`

        Args:
            num (int): count of shallow copies

        Returns:
            Self: new sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl = SortedList("abc")
        new = sl * 3
        assert new == SortedList(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
        ```
        """
        values = self._inner.collapse_lists().repeat(num)
        return self.__class__(values)

    def __rmul__(self, num: int) -> Self:
        return self.__mul__(num)

    def __imul__(self, num: int) -> Self:
        """In-place update of the sorted list with `num` shallow copies of values.

        ``sl.__imul__(num)`` <==> ``sl *= num``

        Runtime complexity: `O(n*log(n))`

        Args:
            num (int): count of shallow copies

        Returns:
            Self: existing sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList

        sl = SortedList("abc")
        sl *= 3
        assert sl == SortedList(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
        ```
        """
        values = self._inner.collapse_lists().repeat(num)
        self.clear()
        self.update(values)
        return self

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T]]]:
        values = self._inner.collapse_lists()
        return (self.__class__, (values,))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted list.

        ``sl.__repr__()`` <==> ``repr(sl)``

        Returns:
            (str): string representation

        """
        return f"{self.__class__.__name__}({list(self)!r})"
