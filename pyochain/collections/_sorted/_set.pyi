# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from types import NotImplementedType
from typing import Any, Final, Self, final, overload, override

from pyochain import SetMut, Vec
from pyochain.abc import PyoIterator, PyoMutableSet, PyoSequence

from ._list import BaseSortedListSet
from ._views import SupportsHashableAndRichComparison

class BaseSortedSet[T: SupportsHashableAndRichComparison](
    PyoMutableSet[T], PyoSequence[T], BaseSortedListSet[T], ABC
):
    set: Final[SetMut[T]]
    @override
    @abstractmethod
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[AbstractSet[T]]]:
        """Support for pickle.

        The tricks played with exposing methods in `SortedSet.__init__` confuse pickle so customize the reducer.

        Returns:
            tuple[type[Self], tuple[AbstractSet[T]]]: tuple of class and arguments

        """

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted set.

        ``ss.__contains__(value)`` <==> ``value in ss``

        Runtime complexity: `O(1)`

        Args:
            value (object): search for value in sorted set

        Returns:
            bool: `True` if `value` in sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert 3 in ss
            assert not 6 in ss
            ```
        """

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | Vec[T]:
        """Lookup value at `index` in sorted set.

        ``ss.__getitem__(index)`` <==> ``ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.


        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            T | Vec[T]: value or list of values

        Examples:
            ```python
            from pyochain.collections import SortedSet
            from pyochain import Vec

            ss = SortedSet("abcde")
            assert ss[2] == "c"
            assert ss[-1] == "e"
            assert ss[2:5] == Vec("c", "d", "e")
            ```
        """

    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted set.

        ``ss.__delitem__(index)`` <==> ``del ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Examples:
            ```python
            from pyochain.collections import SortedSet
            import pytest

            ss = SortedSet("abcde")
            del ss[2]
            assert ss == SortedSet(["a", "b", "d", "e"])
            del ss[:2]
            assert ss == SortedSet(["d", "e"])
            with pytest.raises(IndexError):
                del ss[10]
            ```
        """

    @override
    def __eq__(self, other: object) -> bool | NotImplementedType: ...
    @override
    def __ne__(self, other: object) -> bool | NotImplementedType: ...
    @override
    def __lt__(
        self, other: AbstractSet[object] | Self | object
    ) -> bool | NotImplementedType: ...
    @override
    def __gt__(
        self, other: AbstractSet[object] | Self | object
    ) -> bool | NotImplementedType: ...
    @override
    def __le__(
        self, other: AbstractSet[object] | Self | object
    ) -> bool | NotImplementedType: ...
    @override
    def __ge__(
        self, other: AbstractSet[object] | Self | object
    ) -> bool | NotImplementedType: ...
    @override
    def __len__(self) -> int:
        """Return the size of the sorted set.

        ``ss.__len__()`` <==> ``len(ss)``

        Returns:
            int: size of sorted set

        """

    @override
    def __iter__(self) -> PyoIterator[T]:
        """Return an iterator over the sorted set.

        ``ss.__iter__()`` <==> ``iter(ss)``

        Iterating the sorted set while adding or deleting values may raise a `RuntimeError` or fail to iterate over all values.

        """

    @override
    def __reversed__(self) -> PyoIterator[T]:
        """Return a reverse iterator over the sorted set.

        ``ss.__reversed__()`` <==> ``reversed(ss)``

        Iterating the sorted set while adding or deleting values may raise a `RuntimeError` or fail to iterate over all values.

        """

    def __copy__(self) -> Self: ...
    @override
    def __sub__(self, other: Iterable[Any]) -> Self: ...
    @override
    def __isub__(self, other: Iterable[Any]) -> Self: ...
    @override
    def __and__(self, other: Iterable[Any]) -> Self: ...
    def __rand__(self, other: Iterable[Any]) -> Self: ...
    @override
    def __iand__(self, other: Iterable[Any]) -> Self: ...
    @override
    # pyrefly: ignore [bad-override]
    def __xor__(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def __rxor__(self, other: Iterable[T]) -> Self: ...
    @override
    def __ixor__(self, other: Iterable[T]) -> Self: ...
    @override
    # pyrefly: ignore [bad-override]
    def __or__(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    def __ror__(self, other: Iterable[T]) -> Self: ...
    @override
    def __ior__(self, other: Iterable[T]) -> Self: ...
    @override
    @abstractmethod
    # pyrefly: ignore [bad-override]
    def union(self, *iterables: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return new sorted set with values from itself and all `iterables`.

        The `union` method also corresponds to operator ``|``.

        ``ss.__or__(iterable)`` <==> ``ss | iterable``

        Args:
            *iterables (Iterable[T]): iterable arguments

        Returns:
            Self: new sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert ss.union([4, 5, 6, 7]) == SortedSet([1, 2, 3, 4, 5, 6, 7])
            ```

        """

    @override
    def reset(self, load: int) -> None: ...
    @override
    def bisect_left(self, value: T) -> int: ...
    @override
    def bisect_right(self, value: T) -> int: ...
    @override
    def index(
        self, value: T, start: int | None = None, stop: int | None = None
    ) -> int: ...
    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
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
    def is_disjoint(self, other: Iterable[object]) -> bool: ...
    @override
    def is_subset(self, other: Iterable[object]) -> bool: ...
    @override
    def is_superset(self, other: Iterable[object]) -> bool: ...
    @override
    def add(self, value: T) -> None:
        """Add `value` to sorted set.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): value to add to sorted set
        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet()
            ss.add(3)
            ss.add(1)
            ss.add(2)
            assert ss == SortedSet([1, 2, 3])
            ```

        """

    @override
    def clear(self) -> None:
        """Remove all values from sorted set.

        Runtime complexity: `O(n)`

        """

    @override
    def copy(self) -> Self:
        """Return a shallow copy of the sorted set.

        Runtime complexity: `O(n)`

        Returns:
            Self: new sorted set

        """

    @override
    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted set.

        Runtime complexity: `O(1)`

        Args:
            value (T): value to count in sorted set

        Returns:
            int: number of occurrences of `value` in the sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert ss.count(3) == 1
            ```

        """

    @override
    def discard(self, value: T) -> None:
        """Remove `value` from sorted set if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): `value` to discard from sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.discard(5)
            ss.discard(0)
            assert ss == set([1, 2, 3, 4])
            ```
        """

    @override
    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted set.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int): index of value (default -1)

        Returns:
            T: value at `index`

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet("abcde")
            assert ss.pop() == "e"
            assert ss.pop(2) == "c"
            assert ss == SortedSet(["a", "b", "d"])
            ```
        """

    @override
    def remove(self, value: T) -> None:
        """Remove `value` from sorted set; `value` must be a member.

        If `value` is not a member, raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            value (T): `value` to remove from sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet
            import pytest

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.remove(5)
            assert ss == set([1, 2, 3, 4])
            with pytest.raises(KeyError):
                ss.remove(0)
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def difference(self, *iterables: Iterable[Any]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the difference of two or more sets as a new sorted set.

        The `difference` method also corresponds to operator ``-``.

        ``ss.__sub__(iterable)`` <==> ``ss - iterable``

        The difference is all values that are in this sorted set but not the other `iterables`.

        Args:
            *iterables (Iterable[Any]): iterable arguments

        Returns:
            Self: new sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert ss.difference([4, 5, 6, 7]) == SortedSet([1, 2, 3])
            ```

        """

    def difference_update(self, *iterables: Iterable[T]) -> Self:
        """Remove all values of `iterables` from this sorted set.

        The `difference_update` method also corresponds to operator ``-=``.

        ``ss.__isub__(iterable)`` <==> ``ss -= iterable``

        Args:
            *iterables (Iterable[T]): iterable arguments

        Returns:
            Self: updated sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.difference_update([4, 5, 6, 7])
            assert ss == SortedSet([1, 2, 3])
            ```
        """

    @override
    # pyrefly: ignore [bad-override]
    def intersection(self, *iterables: Iterable[Any]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the intersection of two or more sets as a new sorted set.

        The `intersection` method also corresponds to operator ``&``.

        ``ss.__and__(iterable)`` <==> ``ss & iterable``

        The intersection is all values that are in this sorted set and each of
        the other `iterables`.

        Args:
            *iterables (Iterable[Any]): iterable arguments

        Returns:
            Self: new sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert ss.intersection([4, 5, 6, 7]) == SortedSet([4, 5])
            ```

        """

    def intersection_update(self, *iterables: Iterable[Any]) -> Self:
        """In-place update of the sorted set with the intersection of `iterables`.

        The `intersection_update` method also corresponds to operator ``&=``.

        ``ss.__iand__(iterable)`` <==> ``ss &= iterable``

        Keep only values found in itself and all `iterables`.

        Args:
            *iterables (Iterable[Any]): iterable arguments

        Returns:
            Self: updated sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.intersection_update([4, 5, 6, 7])
            assert ss == SortedSet([4, 5])
            ```

        """

    @override
    # pyrefly: ignore [bad-override]
    def symmetric_difference(self, other: Iterable[T]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the symmetric difference with `other` as a new sorted set.

        The `symmetric_difference` method also corresponds to operator ``^``.

        ``ss.__xor__(other)`` <==> ``ss ^ other``

        The symmetric difference is all values tha are in exactly one of the
        sets.

        Args:
            other (Iterable[T]): `other` iterable

        Returns:
            Self: new sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            assert ss.symmetric_difference([4, 5, 6, 7]) == SortedSet([1, 2, 3, 6, 7])
            ```

        """

    def symmetric_difference_update(self, other: Iterable[T]) -> Self:
        """In-place update of the sorted set with the symmetric difference with `other`.

        The `symmetric_difference_update` method also corresponds to operator
        ``^=``.

        ``ss.__ixor__(other)`` <==> ``ss ^= other``

        Keep only values found in exactly one of itself and `other`.

        Args:
            other (Iterable[T]): `other` iterable

        Returns:
            Self: updated sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.symmetric_difference_update([4, 5, 6, 7])
            assert ss == SortedSet([1, 2, 3, 6, 7])
            ```
        """

    def update(self, *iterables: Iterable[T]) -> Self:
        """In-place update of the sorted set, adding values from all `iterables`.

        The `update` method also corresponds to operator ``|=``.

        ``ss.__ior__(iterable)`` <==> ``ss |= iterable``


        Args:
            *iterables (Iterable[T]): iterable arguments

        Returns:
            Self: updated sorted set

        Examples:
            ```python
            from pyochain.collections import SortedSet

            ss = SortedSet([1, 2, 3, 4, 5])
            ss.update([4, 5, 6, 7])
            assert ss == SortedSet([1, 2, 3, 4, 5, 6, 7])
            ```
        """

@final
class SortedSet[T: SupportsHashableAndRichComparison](BaseSortedSet[T]):
    """Sorted set is a `MutableSet` whose values are maintained in sorted order.

    The design of sorted set is simple: sorted set uses a set for set-operations and maintains a sorted list of values.

    Sorted set values must be hashable and comparable.

    The hash and total ordering of values must not change while they are stored in the sorted set.

    Sorted set comparisons use subset and superset relations.

    Two sorted sets are equal if and only if every element of each sorted set is contained in the other (each is a subset of the other).

    A sorted set is less than another sorted set if and only if the first sorted set is a proper subset
    of the second sorted set (is a subset, but is not equal).

    A sorted set is greater than another sorted set if and only if the first sorted set is a proper superset of the second sorted set (is a superset, but is not equal).

    Optional `iterable` argument provides an initial iterable of values to initialize the sorted set.

    Runtime complexity: `O(n*log(n))`

    ```python
    from pyochain.collections import SortedSet

    ss = SortedSet([3, 1, 2, 5, 4])
    assert repr(ss) == "SortedSet([1, 2, 3, 4, 5])"
    ```

    Args:
        iterable (Iterable[T] | None): initial values (optional)

    """

    def __init__(self, iterable: Iterable[T] | None = None) -> None: ...
    @override
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[AbstractSet[T]]]: ...
    @override
    def union(self, *iterables: Iterable[T]) -> Self: ...
