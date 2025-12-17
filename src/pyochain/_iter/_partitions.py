from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import cytoolz as cz

from .._core import IterWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._main import Iter


class BasePartitions[T](IterWrapper[T]):
    @overload
    def map_windows[R](
        self, length: Literal[1], func: Callable[[tuple[T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[2], func: Callable[[tuple[T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[3], func: Callable[[tuple[T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[4], func: Callable[[tuple[T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[5], func: Callable[[tuple[T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: int, func: Callable[[tuple[T, ...]], R]
    ) -> Iter[R]: ...
    def map_windows[R](
        self, length: int, func: Callable[[tuple[Any, ...]], R]
    ) -> Iter[R]:
        """Calls the given **func** for each contiguous window of size **length** over **self**.

        The windows during mapping overlaps.

        The provided function must have a signature matching the length of the window.

        Args:
            length (int): The length of each window.
            func (Callable[[tuple[Any, ...]], R]): Function to apply to each window.

        Returns:
            Iter[R]: An iterator over the outputs of func.

        Example:
        ```python
        >>> import pyochain as pc

        >>> pc.Iter("abcd").map_windows(2, lambda xy: f"{xy[0]}+{xy[1]}").collect()
        Seq('a+b', 'b+c', 'c+d')
        >>> pc.Iter([1, 2, 3, 4]).map_windows(2, lambda xy: xy).collect()
        Seq((1, 2), (2, 3), (3, 4))
        >>> def moving_average(seq: tuple[int, ...]) -> float:
        ...     return float(sum(seq)) / len(seq)
        >>> pc.Iter([1, 2, 3, 4]).map_windows(2, moving_average).collect()
        Seq(1.5, 2.5, 3.5)

        ```
        """

        def _map_windows(data: Iterable[T]) -> Iterator[R]:
            return map(func, cz.itertoolz.sliding_window(length, data))

        return self._lazy(_map_windows)

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
        Seq((1, 2), (3, 4))

        ```
        If the length of seq is not evenly divisible by n, the final tuple is dropped if pad is not specified, or filled to length n by pad:
        ```python
        >>> pc.Iter([1, 2, 3, 4, 5]).partition(2).collect()
        Seq((1, 2), (3, 4), (5, None))

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
        Seq((1, 2), (3, 4))
        >>> pc.Iter([1, 2, 3, 4, 5]).partition_all(2).collect()
        Seq((1, 2), (3, 4), (5,))

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
        >>> pc.Iter("I have space").partition_by(lambda c: c == " ").collect()
        Seq(('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e'))
        >>>
        >>> data = [1, 2, 1, 99, 88, 33, 99, -1, 5]
        >>> pc.Iter(data).partition_by(lambda x: x > 10).collect()
        Seq((1, 2, 1), (99, 88, 33, 99), (-1, 5))

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
        Seq(('A', 'B', 'C'), ('D', 'E', 'F'), ('G',))

        ```
        """
        return self._lazy(itertools.batched, n)
