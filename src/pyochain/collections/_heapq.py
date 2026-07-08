from __future__ import annotations

import heapq as hq
from abc import ABC, abstractmethod
from heapq import (
    _heapify_max as heapify_max,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
)
from typing import TYPE_CHECKING, Self, SupportsIndex, overload, override

from pyochain import Vec
from pyochain.abc import PyoIterator, PyoMutableSequence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import SupportsRichComparison


class Heap[T: SupportsRichComparison](PyoMutableSequence[T], ABC):  # noqa: PLW1641
    """Abstract base class for heaps."""

    _inner: list[T]

    def __init__(self, data: list[T]) -> None:
        """Transform a `list` into a heap, in-place, in O(len(x)) time."""
        self._inner = data
        self._heapify(self._inner)

    @abstractmethod
    def _heapify(self, data: list[T]) -> None: ...

    @classmethod
    def from_ref(cls, data: list[T]) -> Self:
        """Create a `Heap` instance from an existing `list` without copying.

        Assumes that the provided list already satisfies the corresponding heap invariant.

        Args:
            data (list[T]): A `list` that already satisfies the corresponding heap invariant.
            data (list[T]): A `list` that already satisfies the corresponding heap invariant.

        Returns:
            Heap[T]: A new `Heap` instance.
        """
        instance = cls.__new__(cls)
        instance._inner = data
        return instance

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner!r})"

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, s: slice[SupportsIndex | None], /) -> list[T]: ...
    @override
    def __getitem__(
        self, index: SupportsIndex | slice[SupportsIndex | None]
    ) -> T | list[T]:
        return self._inner[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        # pyrefly: ignore[no-matching-overload]
        return self._inner.__setitem__(index, value)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType]

    @override
    def __delitem__(self, index: int | slice) -> None:
        del self._inner[index]

    @override
    def __eq__(self, other: object) -> bool:
        match other:
            case Heap() | Vec():
                return self.inner == other.inner  # pyright: ignore[reportUnknownMemberType]
            case list():
                return self._inner == other
            case _:
                return NotImplemented

    @override
    def insert(self, index: int, value: T) -> None:
        self._inner.insert(index, value)

    @abstractmethod
    def push(self, item: T) -> None:
        """Push item onto heap, maintaining the heap invariant."""

    @override
    def pop(self, _index: int = -1, /) -> T:
        """Pop the smallest item off the heap, maintaining the heap invariant.

        Warning:
            *index* is kept to maintain compatibility with the `collections.abc.MutableSequence` interface, but it is ignored.

            The smallest item is always popped.

        Returns:
            T: The smallest item from the heap.
        """
        return hq.heappop(self._inner)

    @abstractmethod
    def replace(self, item: T) -> T:
        """Pop and return the current smallest value, and add the new item.

        This is more efficient than `pop()` followed by `push()`, and can be
        more appropriate when using a fixed-size heap.

        Note that the value returned may be larger than item!

        That constrains reasonable uses of this routine unless written as part of a conditional replacement:

        Example:
            ```python
            >>> heap = HeapMin([1, 2, 3])
            >>> item = 4
            >>> item = heap.replace(item) if item > heap[0] else item
            >>> heap
            HeapMin([2, 4, 3])

            ```

        Returns:
            T: The smallest item from the heap.
        """

    @abstractmethod
    def push_pop(self, item: T) -> T:
        """Fast version of a heappush followed by a heappop.

        Returns:
            T: The smallest item from the `Heap`.
        """

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

        >>> base = [1, 3, 5, 7]
        >>> HeapMin(base).merge([0, 2, 4, 8], [5, 10, 15, 20], [], [25]).collect(list)
        [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

        If *key* is not `None`, applies a key function to each element to determine its sort order.

        >>> others = ["cat", "fish", "kangaroo"]
        >>> HeapMin(["dog", "horse"]).merge(others, key=len).collect(list)
        ['dog', 'cat', 'fish', 'horse', 'kangaroo']

        Returns:
            PyoIterator[T | S]: A generator that yields the merged sorted elements from the input iterables.

        """
        from pyochain import Iter

        return Iter(hq.merge(self._inner, *others, key=key, reverse=reverse))

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

        Returns:
            Self: A new `Heap` instance containing the *n* smallest elements from the heap.
        """
        return self.from_ref(hq.nsmallest(n, self._inner, key=key))

    @overload
    def n_largest(self, n: int, key: Callable[[T], SupportsRichComparison]) -> Self: ...
    @overload
    def n_largest(self, n: int, key: None = None) -> Self: ...
    def n_largest(
        self, n: int, key: Callable[[T], SupportsRichComparison] | None = None
    ) -> Self:
        """Find the *n* largest elements in a dataset.

        Equivalent to:  `iterator.sort(key=key, reverse=True)[:n]`

        Returns:
            Self: A new `Heap` instance containing the *n* largest elements from the heap.
        """
        return self.from_ref(hq.nlargest(n, self._inner, key=key))

    @property
    def inner(self) -> list[T]:
        """The underlying list representing the heap.

        Returns:
            list[T]: The underlying list representing the heap.
        """
        return self._inner


class HeapMin[T: SupportsRichComparison](Heap[T]):
    """Abstract base class for heaps."""

    @override
    def _heapify(self, data: list[T]) -> None:
        hq.heapify(data)

    @override
    def push(self, item: T) -> None:
        return hq.heappush(self._inner, item)

    @override
    def pop(self, _index: int = -1, /) -> T:
        return hq.heappop(self._inner)

    @override
    def replace(self, item: T) -> T:
        return hq.heapreplace(self._inner, item)

    @override
    def push_pop(self, item: T) -> T:
        return hq.heappushpop(self._inner, item)


class HeapMax[T: SupportsRichComparison](Heap[T]):
    """Abstract base class for heaps."""

    @override
    def _heapify(self, data: list[T]) -> None:
        heapify_max(data)

    @override
    def push(self, item: T) -> None:
        self._inner.append(item)
        self._siftdown(0, self.len() - 1)

    @override
    def pop(self, _index: int = -1, /) -> T:
        lastelt = self._inner.pop()
        if not self.is_empty():
            returnitem = self[0]
            self[0] = lastelt
            self._siftup(0)
            return returnitem
        return lastelt

    @override
    def replace(self, item: T) -> T:
        returnitem = self[0]  # raises appropriate IndexError if heap is empty
        self[0] = item
        self._siftup(0)
        return returnitem

    @override
    def push_pop(self, item: T) -> T:
        if not self.is_empty() and item < self[0]:  # pyright: ignore[reportOperatorIssue]
            item, self[0] = self[0], item
            self._siftup(0)
        return item

    def _siftdown(self, startpos: int, pos: int) -> None:
        newitem = self[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self[parentpos]
            if parent < newitem:  # pyright: ignore[reportOperatorIssue]
                self[pos] = parent
                pos = parentpos
                continue
            break
        self[pos] = newitem

    def _siftup(self, pos: int) -> None:
        endpos = self.len()
        startpos = pos
        newitem = self[pos]
        # Bubble up the larger child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of larger child.
            rightpos = childpos + 1
            if rightpos < endpos and not self[rightpos] < self[childpos]:  # pyright: ignore[reportOperatorIssue]
                childpos = rightpos
            # Move the larger child up.
            self[pos] = self[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self[pos] = newitem
        self._siftdown(startpos, pos)
