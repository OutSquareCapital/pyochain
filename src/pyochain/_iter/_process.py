from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Iterable, Iterator
from functools import partial
from random import Random
from typing import TYPE_CHECKING, Any, NamedTuple

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter


class Peeked[T](NamedTuple):
    values: tuple[T, ...]
    original: Iterator[T]


def _too_short(item_count: int) -> None:
    return mit.raise_(
        ValueError,
        f"Too few items in iterable (got {item_count})",
    )


def _too_long(item_count: int) -> None:
    return mit.raise_(
        ValueError,
        f"Too many items in iterable (got at least {item_count})",
    )


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
        >>> pc.Seq((1, 2)).iter().cycle().take(5).collect()
        Seq((1, 2, 1, 2, 1))

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
        >>> pc.Seq([1, 2]).iter().interpose(0).collect()
        Seq((1, 0, 2))

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
        >>> data = pc.Seq(list(range(100)))
        >>> data.iter().random_sample(0.1).into(list)  # doctest: +SKIP
        [6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95]
        >>> data.iter().random_sample(0.1).into(list)  # doctest: +SKIP
        [6, 44, 54, 61, 69, 94]
        ```
        Providing an integer seed for random_state will result in deterministic sampling.

        Given the same seed it will return the same sample every time.
        ```python
        >>> data.iter().random_sample(0.1, state=2016).into(list)
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
        >>> data.iter().random_sample(0.1, state=2016).into(list)
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]

        ```
        random_state can also be any object with a method random that returns floats between 0.0 and 1.0 (exclusive).
        ```python
        >>> from random import Random
        >>> randobj = Random(2016)
        >>> data.iter().random_sample(0.1, state=randobj).into(list)
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]

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
        >>> pc.Seq((1, 2, 3)).iter().accumulate(lambda a, b: a + b).collect()
        Seq((1, 3, 6))

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
        >>> pc.Seq((2, 3)).iter().insert_left(1).collect()
        Seq((1, 2, 3))

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
        >>> pc.Seq([1, 2, 3]).iter().peek(2, lambda x: print(f"Peeked {len(x)} values: {x}")).collect()
        Peeked 2 values: (1, 2)
        Seq((1, 2, 3))

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
        >>> pc.Seq([1, 3]).iter().merge_sorted([2, 4]).collect()
        Seq((1, 2, 3, 4))

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
        >>> pc.Seq((1, 2)).iter().interleave((3, 4)).collect()
        Seq((1, 3, 2, 4))

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
        >>> pc.Seq((1, 2)).iter().chain((3, 4), [5]).collect()
        Seq((1, 2, 3, 4, 5))

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
        >>> pc.Seq("ABCABC").iter().elements().sort()
        Seq(['A', 'A', 'B', 'B', 'C', 'C'])

        ```
        Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
        ```python
        >>> import math
        >>> data = [2, 2, 3, 3, 3, 17]
        >>> pc.Seq(data).iter().elements().into(math.prod)
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
        >>> pc.Seq([1, 2, 3]).iter().reverse().collect()
        Seq((3, 2, 1))

        ```
        """

        def _reverse(data: Iterable[T]) -> Iterator[T]:
            return reversed(tuple(data))

        return self._lazy(_reverse)

    def is_strictly_n(
        self,
        n: int,
        too_short: Callable[[int], Iterator[T]] | Callable[[int], None] = _too_short,
        too_long: Callable[[int], Iterator[T]] | Callable[[int], None] = _too_long,
    ) -> Iter[T]:
        """Validate that *iterable* has exactly *n* items and return them if it does.

        If it has fewer than *n* items, call function *too_short* with the actual number of items.

        If it has more than *n* items, call function *too_long* with the number `n + 1`.

        Args:
            n (int): The exact number of items expected.
            too_short (Callable[[int], Iterator[T]] | Callable[[int], None]): Function to call if there are too few items.
            too_long (Callable[[int], Iterator[T]] | Callable[[int], None]): Function to call if there are too many items.

        Returns:
            Iter[T]: A new Iterable wrapper with exactly n items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> iterable = ["a", "b", "c", "d"]
        >>> n = 4
        >>> pc.Seq(iterable).iter().is_strictly_n(n).into(list)
        ['a', 'b', 'c', 'd']

        ```
        Note that the returned iterable must be consumed in order for the check to
        be made.

        By default, *too_short* and *too_long* are functions that raise`ValueError`.
        ```python
        >>> pc.Seq("ab").iter().is_strictly_n(3).into(
        ...     list
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: too few items in iterable (got 2)

        >>> pc.Seq("abc").iter().is_strictly_n(2).into(
        ...     list
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: too many items in iterable (got at least 3)

        ```
        You can instead supply functions that do something else.

        *too_short* will be called with the number of items in *iterable*.

        *too_long* will be called with `n + 1`.
        ```python
        >>> def too_short(item_count):
        ...     raise RuntimeError
        >>> pc.Seq("abcd").iter().is_strictly_n(6, too_short=too_short).into(list)
        Traceback (most recent call last):
        ...
        RuntimeError
        >>> def too_long(item_count):
        ...     print("The boss is going to hear about this")
        >>> pc.Seq("abcdef").iter().is_strictly_n(4, too_long=too_long).into(list)
        The boss is going to hear about this
        ['a', 'b', 'c', 'd']

        ```
        """

        def _strictly_n_(iterable: Iterable[T]) -> Generator[T, Any]:
            it = iter(iterable)

            sent = 0
            for item in itertools.islice(it, n):
                yield item
                sent += 1

            if sent < n:
                too_short(sent)
                return

            for _ in it:
                too_long(n + 1)
                return

        return self._lazy(_strictly_n_)
