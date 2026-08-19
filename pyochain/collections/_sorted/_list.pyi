# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import NotImplementedType
from typing import Self, overload, override, type_check_only

from _typeshed import SupportsRichComparison

from pyochain import Vec
from pyochain.abc import PyoIterator, PyoMutableSequence

from ._core import BaseSortedListSet

@type_check_only
class BaseSortedList[T](BaseSortedListSet[T], PyoMutableSequence[T], ABC):
    load: int
    @abstractmethod
    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted list containing all values in both sequences.

        ``sl.__add__(other)`` <==> ``sl + other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(n*log(n))`

        Args:
            other (Iterable[T]): other iterable

        Returns:
            Self: new sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl1 = SortedList("bat")
        sl2 = SortedList("cat")
        added = sl1 + sl2
        assert added == SortedList(["a", "a", "b", "c", "t", "t"])

        skl1 = SortedKeyList([5, 4, 3], key=neg)
        skl2 = SortedKeyList([2, 1, 0], key=neg)
        new = skl1 + skl2
        assert new == [5, 4, 3, 2, 1, 0]
        ```
        """

    @abstractmethod
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
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList("abc")
        new = sl * 3
        assert new == SortedList(["a", "a", "a", "b", "b", "b", "c", "c", "c"])

        skl = SortedKeyList([3, 2, 1], key=neg)
        new = skl * 2
        assert new == [3, 3, 2, 2, 1, 1]
        ```
        """

    @override
    def __iter__(self) -> PyoIterator[T]:
        """Return an iterator over the sorted list.

        ``sl.__iter__()`` <==> ``iter(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted list.

        ``sl.__contains__(value)`` <==> ``value in sl``

        Runtime complexity: `O(log(n))`

        Args:
            value (object): search for value in sorted list

        Returns:
            bool: `True` if `value` in sorted list.

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 3, 4, 5])
        assert 3 in sl

        skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        assert 3 in skl
        ```

        """

    @override
    def __eq__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is equal to `other`.

        ``sl.__eq__(other)`` <==> ``sl == other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            NotImplementedType | bool: true if sorted list is equal to `other`

        """

    @override
    def __ne__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is not equal to `other`.

        ``sl.__ne__(other)`` <==> ``sl != other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            NotImplementedType | bool: true if sorted list is not equal to `other`

        """

    def __lt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than `other`.

        ``sl.__lt__(other)`` <==> ``sl < other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence
        Returns:
            NotImplementedType | bool: true if sorted list is less than `other`

        """

    def __gt__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than `other`.

        ``sl.__gt__(other)`` <==> ``sl > other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            NotImplementedType | bool: true if sorted list is greater than `other`

        """

    def __le__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is less than or equal to `other`.

        ``sl.__le__(other)`` <==> ``sl <= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            NotImplementedType | bool: true if sorted list is less than or equal to `other`

        """

    def __ge__(self, other: object) -> NotImplementedType | bool:
        """Return true if and only if sorted list is greater than or equal to `other`.

        ``sl.__ge__(other)`` <==> ``sl >= other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        Args:
            other (object): `other` sequence

        Returns:
            NotImplementedType | bool: true if sorted list is greater than or equal to `other`

        """

    @override
    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted list.

        ``sl.__delitem__(index)`` <==> ``del sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            index (int | slice): integer or slice for indexing

        Examples:
            ```python
            from pyochain.collections import SortedList

            sl = SortedList("abcde")

            del sl[2]
            assert sl == ["a", "b", "d", "e"]

            del sl[:2]
            assert sl == ["d", "e"]
            ```
        """

    @override
    def __len__(self) -> int:
        """Return the size of the sorted list.

        ``sl.__len__()`` <==> ``len(sl)``

        Returns:
            int: size of sorted list

        """

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | Vec[T]:
        """Lookup value at `index` in sorted list.

        ``sl.__getitem__(index)`` <==> ``sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            T | Vec[T]: value or list of values

        Examples:
        ```python
        from pyochain.collections import SortedList
        from pyochain import Vec

        sl = SortedList("abcde")

        assert sl[1] == "b"
        assert sl[-1] == "e"
        assert sl[2:5] == Vec("c", "d", "e")
        ```
        """

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

    def __radd__(self, other: Iterable[T]) -> Self: ...
    def __rmul__(self, num: int) -> Self: ...
    @override
    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        """Warning: Raise not-implemented error.

        use ``del sl[index]`` and ``sl.add(value)`` instead.

        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``

        Args:
            index (int | slice): integer or slice for indexing
            value (T | Iterable[T]): value or iterable of values
        """

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
    @override
    def add(self, value: T) -> None: ...
    @override
    def append(self, value: T) -> None:
        """Warning: Raise not-implemented error.

        use ``sl.add(value)`` instead

        Implemented to override `MutableSequence.append` which provides an erroneous default implementation.

        Args:
            value (T): value to append to sorted list

        """

    @override
    def discard(self, value: T) -> None: ...
    @override
    def remove(self, value: T, /) -> None: ...
    @override
    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): value to count in sorted list

        Returns:
            int: count of occurrences of `value` in sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList
        from operator import neg

        sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        assert sl.count(3) == 3

        skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)
        assert skl.count(2) == 2
        ```
        """

    @override
    def clear(self) -> None: ...
    @override
    def extend(self, values: object) -> None:
        """Warning: raise not-implemented error.

        use ``sl.update(values)`` instead

        Implemented to override `MutableSequence.extend` which provides an erroneous default implementation.

        Args:
            values (object): values to extend sorted list with
        """

    @override
    def insert(self, index: int, value: T) -> None:
        """Warning: raise not-implemented error.

        use ``sl.add(value)`` instead

        Args:
            index (int): index to insert value at
            value (T): value to insert
        """

    @override
    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted list.

        Raise :exc:`IndexError` if the sorted list is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            index (int): index of value (default -1)

        Returns:
            T: value at `index` in sorted list

        Examples:
        ```python
        from pyochain.collections import SortedList, SortedKeyList

        sl = SortedList("abcde")
        assert sl.pop() == "e"
        assert sl.pop(2) == "c"
        assert sl == SortedList(["a", "b", "d"])
        ```
        """

    @override
    def bisect_left(self, value: T) -> int: ...
    @override
    def bisect_right(self, value: T) -> int: ...
    @override
    def index(
        self, value: T, start: int | None = None, stop: int | None = None
    ) -> int: ...
    @override
    def reset(self, load: int) -> None: ...
    def update(self, iterable: Iterable[T]) -> None:
        """Add all the values from *iterable* to the `SortedCollection`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        Args:
            iterable (Iterable[T]): iterable of values to add

        Examples:
            ```python
            from pyochain.collections import SortedList, SortedKeyList

            sl = SortedList()
            sl.update([3, 1, 2])
            assert sl == SortedList([1, 2, 3])

            from operator import neg

            skl = SortedKeyList(key=neg)
            skl.update([3, 1, 2])
            assert skl == [3, 2, 1]
            ```
        """

    @override
    def reverse(self) -> None:
        """Warning: raise not-implemented error.

        Sorted list maintains values in ascending sort order.

        Values may not be reversed in-place.

        Use ``reversed(sl)`` for an iterator over values in descending sort order.

        Implemented to override `MutableSequence.reverse` which provides an erroneous default implementation.
        """

class SortedList[T: SupportsRichComparison](BaseSortedList[T]):
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

    def __new__(cls, iterable: Iterable[T] | None = None) -> Self: ...
    def __copy__(self) -> Self: ...
    @override
    def __add__(self, other: Iterable[T]) -> Self: ...
    @override
    def __mul__(self, num: int) -> Self: ...
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Vec[T]]]: ...
    @override
    def copy(self) -> Self: ...
