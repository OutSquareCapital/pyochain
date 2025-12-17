from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from random import Random
from typing import TYPE_CHECKING, Any, NamedTuple

import cytoolz as cz

from .._core import IterWrapper

if TYPE_CHECKING:
    from .._results import Result
    from ._main import Iter


class Peeked[T](NamedTuple):
    values: tuple[T, ...]
    original: Iterator[T]


class BaseProcess[T](IterWrapper[T]):
    def cycle(self) -> Iter[T]:
        """Repeat the sequence indefinitely.

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use Iter.take() or Iter.slice() to limit the number of items taken.

        Returns:
            Iter[T]: A new Iterable wrapper that cycles through the elements indefinitely.
        ```python

        Example:
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).cycle().take(5).collect()
        Seq(1, 2, 1, 2, 1)

        ```
        """
        return self._lazy(itertools.cycle)

    def interpose(self, element: T) -> Iter[T]:
        """Interpose element between items and return a new Iterable wrapper.

        Args:
            element (T): The element to interpose between items.

        Returns:
            Iter[T]: A new Iterable wrapper with the element interposed.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).interpose(0).collect()
        Seq(1, 0, 2)

        ```
        """
        return self._lazy(partial(cz.itertoolz.interpose, element))

    def random_sample(
        self,
        probability: float,
        state: Random | int | None = None,
    ) -> Iter[T]:
        """Return elements from a sequence with probability of prob.

        Returns a lazy iterator of random items from seq.

        random_sample considers each item independently and without replacement.

        See below how the first time it returned 13 items and the next time it returned 6 items.

        Args:
            probability (float): The probability of including each element.
            state (Random | int | None): Random state or seed for deterministic sampling.

        Returns:
            Iter[T]: A new Iterable wrapper with randomly sampled elements.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Iter(range(100)).collect()
        >>> data.iter().random_sample(0.1).collect()  # doctest: +SKIP
        Seq(6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95)
        >>> data.iter().random_sample(0.1).collect()  # doctest: +SKIP
        Seq(6, 44, 54, 61, 69, 94)
        ```
        Providing an integer seed for random_state will result in deterministic sampling.

        Given the same seed it will return the same sample every time.
        ```python
        >>> data.iter().random_sample(0.1, state=2016).collect()
        Seq(7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98)
        >>> data.iter().random_sample(0.1, state=2016).collect()
        Seq(7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98)

        ```
        random_state can also be any object with a method random that returns floats between 0.0 and 1.0 (exclusive).
        ```python
        >>> from random import Random
        >>> randobj = Random(2016)
        >>> data.iter().random_sample(0.1, state=randobj).collect()
        Seq(7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98)

        ```
        """
        return self._lazy(
            partial(cz.itertoolz.random_sample, probability, random_state=state),
        )

    def accumulate(self, func: Callable[[T, T], T]) -> Iter[T]:
        """Return cumulative application of binary op provided by the function.

        Args:
            func (Callable[[T, T], T]): A binary function to apply cumulatively.

        Returns:
            Iter[T]: A new Iterable wrapper with accumulated results.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 3)).accumulate(lambda a, b: a + b).collect()
        Seq(1, 3, 6)

        ```
        """
        return self._lazy(partial(cz.itertoolz.accumulate, func))

    def insert_left(self, value: T) -> Iter[T]:
        """Prepend value to the sequence and return a new Iterable wrapper.

        Args:
            value (T): The value to prepend.

        Returns:
            Iter[T]: A new Iterable wrapper with the value prepended.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((2, 3)).insert_left(1).collect()
        Seq(1, 2, 3)

        ```
        """
        return self._lazy(partial(cz.itertoolz.cons, value))

    def peek(self, n: int, func: Callable[[Iterable[T]], Any]) -> Iter[T]:
        """Retrieve the first n items from the iterable, pass them to func, and return the original iterable.

        Allow to pass side-effect functions that process the peeked items without consuming the original Iterator.

        Args:
            n (int): Number of items to peek.
            func (Callable[[Iterable[T]], Any]): Function to process the peeked items.

        Returns:
            Iter[T]: A new Iterable wrapper with the peeked items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).peek(2, lambda x: print(f"Peeked {len(x)} values: {x}")).collect()
        Peeked 2 values: (1, 2)
        Seq(1, 2, 3)

        ```
        """

        def _peek(data: Iterable[T]) -> Iterator[T]:
            peeked = Peeked(*cz.itertoolz.peekn(n, data))
            func(peeked.values)
            return peeked.original

        return self._lazy(_peek)

    def merge_sorted(
        self,
        *others: Iterable[T],
        sort_on: Callable[[T], Any] | None = None,
    ) -> Iter[T]:
        """Merge already-sorted sequences.

        Args:
            *others (Iterable[T]): Other sorted iterables to merge.
            sort_on (Callable[[T], Any] | None): Optional key function for sorting.

        Returns:
            Iter[T]: A new Iterable wrapper with merged sorted elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 3]).merge_sorted([2, 4]).collect()
        Seq(1, 2, 3, 4)

        ```
        """
        return self._lazy(cz.itertoolz.merge_sorted, *others, key=sort_on)

    def interleave(self, *others: Iterable[T]) -> Iter[T]:
        """Interleave multiple sequences element-wise.

        Args:
            *others (Iterable[T]): Other iterables to interleave.

        Returns:
            Iter[T]: A new Iterable wrapper with interleaved elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).interleave((3, 4)).collect()
        Seq(1, 3, 2, 4)

        ```
        """

        def _interleave(data: Iterable[T]) -> Iterator[T]:
            return cz.itertoolz.interleave((data, *others))

        return self._lazy(_interleave)

    def chain(self, *others: Iterable[T]) -> Iter[T]:
        """Concatenate zero or more iterables, any of which may be infinite.

        An infinite sequence will prevent the rest of the arguments from being included.

        We use chain.from_iterable rather than chain(*seqs) so that seqs can be a generator.

        Args:
            *others (Iterable[T]): Other iterables to concatenate.

        Returns:
            Iter[T]: A new Iterable wrapper with concatenated elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).chain((3, 4), [5]).collect()
        Seq(1, 2, 3, 4, 5)

        ```
        """

        def _chain(data: Iterable[T]) -> Iterator[T]:
            return cz.itertoolz.concat((data, *others))

        return self._lazy(_chain)

    def elements(self) -> Iter[T]:
        """Iterator over elements repeating each as many times as its count.

        Note:
            if an element's count has been set to zero or is a negative
            number, elements() will ignore it.

        Returns:
            Iter[T]: A new Iterable wrapper with elements repeated according to their counts.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCABC").elements().sort()
        Seq('A', 'A', 'B', 'B', 'C', 'C')

        ```
        Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
        ```python
        >>> import math
        >>> data = [2, 2, 3, 3, 3, 17]
        >>> pc.Iter(data).elements().into(math.prod)
        1836

        ```
        """
        from collections import Counter

        def _elements(data: Iterable[T]) -> Iterator[T]:
            return Counter(data).elements()

        return self._lazy(_elements)

    def reverse(self) -> Iter[T]:
        """Return a new Iterable wrapper with elements in reverse order.

        The result is a new iterable over the reversed sequence.

        Note:
            This method must consume the entire iterable to perform the reversal.

        Returns:
            Iter[T]: A new Iterable wrapper with elements in reverse order.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).reverse().collect()
        Seq(3, 2, 1)

        ```
        """

        def _reverse(data: Iterable[T]) -> Iterator[T]:
            return reversed(tuple(data))

        return self._lazy(_reverse)

    def is_strictly_n(self, n: int) -> Iter[Result[T, ValueError]]:
        """Yield`Ok[T]` as long as the iterable has exactly *n* items.

        If it has fewer than *n* items, yield `Err[ValueError]` with the actual number of items.

        If it has more than *n* items, yield `Err[ValueError]` with the number `n + 1`.

        Note that the returned iterable must be consumed in order for the check to
        be made.

        Args:
            n (int): The exact number of items expected.

        Returns:
            Iter[Result[T, ValueError]]: A new Iterable wrapper yielding results based on the item count.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = ["a", "b", "c", "d"]
        >>> n = 4
        >>> pc.Iter(data).is_strictly_n(n).collect()
        Seq(Ok(value='a'), Ok(value='b'), Ok(value='c'), Ok(value='d'))
        >>> pc.Iter("ab").is_strictly_n(3).collect()  # doctest: +NORMALIZE_WHITESPACE
        Seq(Ok(value='a'), Ok(value='b'),
        Err(error=ValueError('Too few items in iterable (got 2)')))
        >>> pc.Iter("abc").is_strictly_n(2).collect()  # doctest: +NORMALIZE_WHITESPACE
        Seq(Ok(value='a'), Ok(value='b'),
        Err(error=ValueError('Too many items in iterable (got at least 3)')))

        ```
        You can easily combine this with `.map(lambda r: r.map_err(...))` to handle the errors as you wish.
        ```python
        >>> def _my_err(e: ValueError) -> str:
        ...     return f"custom error: {e}"
        >>>
        >>> pc.Iter([1]).is_strictly_n(0).map(lambda r: r.map_err(_my_err)).collect()
        Seq(Err(error='custom error: Too many items in iterable (got at least 1)'),)

        ```
        Or use `.filter_map(...)` to only keep the `Ok` values.
        ```python
        >>> pc.Iter([1, 2, 3]).is_strictly_n(2).filter_map(lambda r: r.ok()).collect()
        Seq(1, 2)

        ```
        """
        from .._results import Err, Ok

        def _strictly_n_(iterable: Iterable[T]) -> Iterator[Result[T, ValueError]]:
            it = iter(iterable)

            sent = 0
            for item in itertools.islice(it, n):
                yield Ok(item)
                sent += 1

            if sent < n:
                e = ValueError(f"Too few items in iterable (got {sent})")
                yield Err(e)

            for _ in it:
                e = ValueError(f"Too many items in iterable (got at least {n + 1})")
                yield Err(e)

        return self._lazy(_strictly_n_)
