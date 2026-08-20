from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Self, SupportsIndex, overload, override

from _typeshed import SupportsRichComparison

from pyochain import Vec
from pyochain.abc import PyoIterator, PyoMutableSequence

class Heap[T: SupportsRichComparison](PyoMutableSequence[T], ABC):
    """Abstract base class for heaps."""

    def __new__(cls, data: Iterable[T]) -> Self: ...
    @override
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, s: slice[SupportsIndex | None], /) -> Vec[T]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | Vec[T]: ...
    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None: ...
    @override
    def __delitem__(self, index: int | slice) -> None: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    @staticmethod
    def from_ref[S: SupportsRichComparison](data: list[S]) -> Heap[S]:
        """Create a `Heap` instance from an existing `list` without copying.

        Assumes that the provided list already satisfies the corresponding heap invariant.

        Args:
            data (list[S]): A `list` that already satisfies the corresponding heap invariant.

        Returns:
            Heap[S]: A new `Heap` instance.
        """

    @abstractmethod
    def push(self, item: T) -> None:
        """Push item onto heap, maintaining the heap invariant."""

    @abstractmethod
    def replace(self, item: T) -> T:
        """Pop and return the current smallest value, and add the new item.

        This is more efficient than `pop()` followed by `push()`, and can be
        more appropriate when using a fixed-size heap.

        Note that the value returned may be larger than item!

        That constrains reasonable uses of this routine unless written as part of a conditional replacement:

        Example:
            ```python
            from pyochain.collections import HeapMin

            heap = HeapMin([1, 2, 3])
            item = 4
            item = heap.replace(item) if item > heap[0] else item
            assert heap == HeapMin([2, 4, 3])
            ```

        Args:
            item (T): The new item to be added to the heap.

        Returns:
            T: The smallest item from the heap.
        """

    @abstractmethod
    def push_pop(self, item: T) -> T:
        """Fast version of a heappush followed by a heappop.

        Args:
            item (T): The new item to be added to the heap.

        Returns:
            T: The smallest item from the `Heap`.
        """

    @abstractmethod
    @override
    def pop(self, _index: int = -1, /) -> T:
        """Pop the smallest item off the heap, maintaining the heap invariant.

        Warning:
            *index* is kept to maintain compatibility with the `collections.abc.MutableSequence` interface, but it is ignored.

            The smallest item is always popped.

        Args:
            _index (int): Ignored.

        Returns:
            T: The smallest item from the heap.
        """

    @override
    def insert(self, index: SupportsIndex, value: T) -> None: ...
    def merge[S: SupportsRichComparison](
        self,
        *others: Iterable[S],
        key: Callable[[T | S], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> PyoIterator[T | S]:
        """Merge *self* and *others* into a single sorted output.

        Similar to `vec.iter().chain(*others).sort()`, but:

        - returns an `Iterator`
        - does not pull the data into memory all at once
        - assumes that each of the input streams is already sorted (smallest to largest).

        ```python
        from pyochain.collections import HeapMin

        base = [1, 3, 5, 7]
        x = HeapMin(base).merge([0, 2, 4, 8], [5, 10, 15, 20], [], [25]).collect(list)
        assert x == [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]
        ```

        If *key* is not `None`, applies a key function to each element to determine its sort order.

        ```python
        others = ["cat", "fish", "kangaroo"]
        x = HeapMin(["dog", "horse"]).merge(others, key=len).collect(list)
        assert x == ["dog", "cat", "fish", "horse", "kangaroo"]
        ```

        Args:
            *others (Iterable[S]): Other sorted iterables to merge with *self*.
            key (Callable[[T | S], SupportsRichComparison] | None): A function that extracts a comparison key from each element. Defaults to `None`.
            reverse (bool): If `True`, the merged output will be in descending order. Defaults to `False`.

        Returns:
            PyoIterator[T | S]: A generator that yields the merged sorted elements from the input iterables.

        """

    @overload
    def n_smallest(
        self, n: int, key: Callable[[T], SupportsRichComparison]
    ) -> Self: ...
    @overload
    def n_smallest(self, n: int, key: None = None) -> Self: ...
    def n_smallest(
        self, n: int, key: Callable[[T], SupportsRichComparison] | None = None
    ) -> Self:
        """Find the *n* smallest elements in a dataset.

        Equivalent to:  `iterator.sort(key=key)[:n]`

        Args:
            n (int): The number of smallest elements to retrieve.
            key (Callable[[T], SupportsRichComparison] | None): A function that extracts a comparison key from each element. Defaults to `None`.

        Returns:
            Self: A new `Heap` instance containing the *n* smallest elements from the heap.
        """

    @overload
    def n_largest(self, n: int, key: Callable[[T], SupportsRichComparison]) -> Self: ...
    @overload
    def n_largest(self, n: int, key: None = None) -> Self: ...
    def n_largest(
        self, n: int, key: Callable[[T], SupportsRichComparison] | None = None
    ) -> Self:
        """Find the *n* largest elements in a dataset.

        Equivalent to:  `iterator.sort(key=key, reverse=True)[:n]`

        Args:
            n (int): The number of largest elements to retrieve.
            key (Callable[[T], SupportsRichComparison] | None): A function that extracts a comparison key from each element. Defaults to `None`.

        Returns:
            Self: A new `Heap` instance containing the *n* largest elements from the heap.
        """

class HeapMin[T: SupportsRichComparison](Heap[T]):
    """Heap implementation that maintains the smallest element at the top."""
    @override
    @staticmethod
    def from_ref[S: SupportsRichComparison](data: list[S]) -> HeapMin[S]: ...
    @override
    def push(self, item: T) -> None: ...
    @override
    def pop(self, _index: int = -1, /) -> T: ...
    @override
    def replace(self, item: T) -> T: ...
    @override
    def push_pop(self, item: T) -> T: ...

class HeapMax[T: SupportsRichComparison](Heap[T]):
    """Heap implementation that maintains the largest element at the top."""
    @override
    @staticmethod
    def from_ref[S: SupportsRichComparison](data: list[S]) -> HeapMax[S]: ...
    @override
    def push(self, item: T) -> None: ...
    @override
    def pop(self, _index: int = -1, /) -> T: ...
    @override
    def replace(self, item: T) -> T: ...
    @override
    def push_pop(self, item: T) -> T: ...
