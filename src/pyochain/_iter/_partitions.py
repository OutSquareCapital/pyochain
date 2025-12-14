from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Literal, overload

import cytoolz as cz

from .._core import IterWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._main import Iter


class BasePartitions[T](IterWrapper[T]):
    @overload
    def windows(self, length: Literal[1]) -> Iter[tuple[T]]: ...
    @overload
    def windows(self, length: Literal[2]) -> Iter[tuple[T, T]]: ...
    @overload
    def windows(self, length: Literal[3]) -> Iter[tuple[T, T, T]]: ...
    @overload
    def windows(self, length: Literal[4]) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def windows(self, length: Literal[5]) -> Iter[tuple[T, T, T, T, T]]: ...

    def windows(self, length: int) -> Iter[tuple[T, ...]]:
        """A sequence of overlapping subsequences of the given length.

        This function allows you to apply custom function not available in the rolling namespace.

        Args:
            length (int): The length of each window.

        Returns:
            Iter[tuple[T, ...]]: An iterable of overlapping subsequences.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 4]).iter().windows(2).collect()
        Seq(((1, 2), (2, 3), (3, 4)))
        >>> def moving_average(seq: tuple[int, ...]) -> float:
        ...     return float(sum(seq)) / len(seq)
        >>> pc.Seq([1, 2, 3, 4]).iter().windows(2).map(moving_average).collect()
        Seq((1.5, 2.5, 3.5))

        ```
        """
        return self._lazy(partial(cz.itertoolz.sliding_window, length))

    @overload
    def partition(self, n: Literal[1], pad: None = None) -> Iter[tuple[T]]: ...
    @overload
    def partition(self, n: Literal[2], pad: None = None) -> Iter[tuple[T, T]]: ...
    @overload
    def partition(self, n: Literal[3], pad: None = None) -> Iter[tuple[T, T, T]]: ...
    @overload
    def partition(self, n: Literal[4], pad: None = None) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def partition(
        self,
        n: Literal[5],
        pad: None = None,
    ) -> Iter[tuple[T, T, T, T, T]]: ...
    @overload
    def partition(self, n: int, pad: int) -> Iter[tuple[T, ...]]: ...
    def partition(self, n: int, pad: int | None = None) -> Iter[tuple[T, ...]]:
        """Partition sequence into tuples of length n.

        Args:
            n (int): Length of each partition.
            pad (int | None): Value to pad the last partition if needed. Defaults to None.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3, 4]).partition(2).collect()
        Seq(((1, 2), (3, 4)))

        ```
        If the length of seq is not evenly divisible by n, the final tuple is dropped if pad is not specified, or filled to length n by pad:
        ```python
        >>> pc.Iter([1, 2, 3, 4, 5]).partition(2).collect()
        Seq(((1, 2), (3, 4), (5, None)))

        ```
        """
        return self._lazy(partial(cz.itertoolz.partition, n, pad=pad))

    def partition_all(self, n: int) -> Iter[tuple[T, ...]]:
        """Partition all elements of sequence into tuples of length at most n.

        The final tuple may be shorter to accommodate extra elements.

        Args:
            n (int): Maximum length of each partition.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3, 4]).partition_all(2).collect()
        Seq(((1, 2), (3, 4)))
        >>> pc.Iter([1, 2, 3, 4, 5]).partition_all(2).collect()
        Seq(((1, 2), (3, 4), (5,)))

        ```
        """
        return self._lazy(partial(cz.itertoolz.partition_all, n))

    def partition_by(self, predicate: Callable[[T], bool]) -> Iter[tuple[T, ...]]:
        """Partition the `iterable` into a sequence of `tuples` according to a predicate function.

        Every time the output of `predicate` changes, a new `tuple` is started,
        and subsequent items are collected into that `tuple`.

        Args:
            predicate (Callable[[T], bool]): Function to determine partition boundaries.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        >>> import pyochain as pc
        >>> pc.Seq("I have space").iter().partition_by(lambda c: c == " ").collect()
        Seq((('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')))
        >>>
        >>> data = [1, 2, 1, 99, 88, 33, 99, -1, 5]
        >>> pc.Seq(data).iter().partition_by(lambda x: x > 10).collect()
        Seq(((1, 2, 1), (99, 88, 33, 99), (-1, 5)))

        ```
        """
        return self._lazy(partial(cz.recipes.partitionby, predicate))

    def batch(self, n: int) -> Iter[tuple[T, ...]]:
        """Batch elements into tuples of length n and return a new Iter.

        - The last batch may be shorter than n.
        - The data is consumed lazily, just enough to fill a batch.
        - The result is yielded as soon as a batch is full or when the input iterable is exhausted.

        Args:
            n (int): Number of elements in each batch.

        Returns:
            Iter[tuple[T, ...]]: An iterable of batched tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCDEFG").batch(3).collect()
        Seq((('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)))

        ```
        """
        return self._lazy(itertools.batched, n)
