from __future__ import annotations

import functools
import statistics
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, NamedTuple

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter


class Unzipped[T, V](NamedTuple):
    first: Iter[T]
    second: Iter[V]


class BaseAgg[T](IterWrapper[T]):
    def join(self: IterWrapper[str], sep: str) -> str:
        """Join all elements of the `Iterable` into a single `string`, with a specified separator.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["a", "b", "c"]).join("-")
        'a-b-c'

        ```
        """
        return self.into(functools.partial(str.join, sep))

    def unzip[U, V](self: IterWrapper[tuple[U, V]]) -> Unzipped[U, V]:
        """Converts an iterator of pairs into a pair of iterators.

        Returns:
            Unzipped[U, V]: NamedTuple with first and second iterators.

        `Iter.unzip()` consumes the iterator of pairs.

        Returns an Unzipped NamedTuple, containing two iterators:

        - one from the left elements of the pairs
        - one from the right elements.

        This function is, in some sense, the opposite of zip.
        ```python
        >>> import pyochain as pc
        >>> data = [(1, "a"), (2, "b"), (3, "c")]
        >>> unzipped = pc.Seq(data).unzip()
        >>> unzipped.first.collect()
        Seq((1, 2, 3))
        >>> unzipped.second.collect()
        Seq(('a', 'b', 'c'))

        ```
        """
        from ._main import Iter

        def _unzip(data: Iterable[tuple[U, V]]) -> Unzipped[U, V]:
            d: tuple[tuple[U, V], ...] = tuple(data)
            return Unzipped(Iter(x[0] for x in d), Iter(x[1] for x in d))

        return self.into(_unzip)

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Apply a function of two arguments cumulatively to the items of an iterable, from left to right.

        Args:
            func (Callable[[T, T], T]): Function to apply cumulatively to the items of the iterable.

        Returns:
            T: Single value resulting from cumulative reduction.

        This effectively reduces the iterable to a single value.

        If initial is present, it is placed before the items of the iterable in the calculation.

        It then serves as a default when the iterable is empty.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).reduce(lambda a, b: a + b)
        6

        ```
        """

        def _reduce(data: Iterable[T]) -> T:
            return functools.reduce(func, data)

        return self.into(_reduce)

    def combination_index(self, r: Iterable[T]) -> int:
        """Equivalent to list(combinations(iterable, r)).index(element).

        Args:
            r (Iterable[T]): The combination to find the index of.

        Returns:
            int: The index of the combination.

        The subsequences of iterable that are of length r can be ordered lexicographically.

        combination_index computes the index of the first element, without computing the previous combinations.

        ValueError will be raised if the given element isn't one of the combinations of iterable.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq("abcdefg").combination_index("adf")
        10

        ```
        """
        return self.into(functools.partial(mit.combination_index, r))

    def first(self) -> T:
        """Return the first element.

        Returns:
            T: The first element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9]).first()
        9

        ```
        """
        return self.into(cz.itertoolz.first)

    def second(self) -> T:
        """Return the second element.

        Returns:
            T: The second element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9, 8]).second()
        8

        ```
        """
        return self.into(cz.itertoolz.second)

    def last(self) -> T:
        """Return the last element.

        Returns:
            T: The last element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([7, 8, 9]).last()
        9

        ```
        """
        return self.into(cz.itertoolz.last)

    def length(self) -> int:
        """Return the length of the Iterable.

        Like the builtin len but works on lazy sequences.

        Returns:
            int: The count of elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).length()
        2

        ```
        """
        return self.into(cz.itertoolz.count)

    def item(self, index: int) -> T:
        """Return item at index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            T: The item at the specified index.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([10, 20]).item(1)
        20

        ```
        """
        return self.into(functools.partial(cz.itertoolz.nth, index))

    def argmax[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a maximum value in an iterable.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq("abcdefghabcd").argmax()
        7
        >>> pc.Seq([0, 1, 2, 3, 3, 2, 1, 0]).argmax()
        3

        ```
        For example, identify the best machine learning model:
        ```python
        >>> models = pc.Seq(["svm", "random forest", "knn", "naïve bayes"])
        >>> accuracy = pc.Seq([68, 61, 84, 72])
        >>> # Most accurate model
        >>> models.item(accuracy.argmax())
        'knn'
        >>>
        >>> # Best accuracy
        >>> accuracy.into(max)
        84

        ```
        """
        return self.into(mit.argmax, key=key)

    def argmin[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a minimum value in an iterable.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the minimum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq("efghabcdijkl").argmin()
        4
        >>> pc.Seq([3, 2, 1, 0, 4, 2, 1, 0]).argmin()
        3

        ```

        For example, look up a label corresponding to the position of a value that minimizes a cost function:
        ```python
        >>> def cost(x):
        ...     "Days for a wound to heal given a subject's age."
        ...     return x**2 - 20 * x + 150
        >>> labels = pc.Seq(["homer", "marge", "bart", "lisa", "maggie"])
        >>> ages = pc.Seq([35, 30, 10, 9, 1])
        >>> # Fastest healing family member
        >>> labels.item(ages.argmin(key=cost))
        'bart'
        >>> # Age with fastest healing
        >>> ages.into(min, key=cost)
        10

        ```
        """
        return self.into(mit.argmin, key=key)

    def sum[U: int | float](self: IterWrapper[U]) -> U | Literal[0]:
        """Return the sum of the sequence.

        Returns:
            U | Literal[0]: The sum of all elements.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).sum()
        6

        ```
        """
        return self.into(sum)

    def min[U: int | float](self: IterWrapper[U]) -> U:
        """Return the minimum of the sequence.

        Returns:
            U: The minimum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).min()
        1

        ```
        """
        return self.into(min)

    def max[U: int | float](self: IterWrapper[U]) -> U:
        """Return the maximum of the sequence.

        Returns:
            U: The maximum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).max()
        3

        ```
        """
        return self.into(max)

    def mean[U: int | float](self: IterWrapper[U]) -> float:
        """Return the mean of the sequence.

        Returns:
            float: The mean value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).mean()
        2

        ```
        """
        return self.into(statistics.mean)

    def median[U: int | float](self: IterWrapper[U]) -> float:
        """Return the median of the sequence.

        Returns:
            float: The median value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 3, 2]).median()
        2

        ```
        """
        return self.into(statistics.median)

    def mode[U: int | float](self: IterWrapper[U]) -> U:
        """Return the mode of the sequence.

        Returns:
            U: The mode value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 2, 3]).mode()
        2

        ```
        """
        return self.into(statistics.mode)

    def stdev[U: int | float](
        self: IterWrapper[U],
    ) -> float:
        """Return the standard deviation of the sequence.

        Returns:
            float: The standard deviation.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).stdev()
        1.0

        ```
        """
        return self.into(statistics.stdev)

    def variance[U: int | float](
        self: IterWrapper[U],
    ) -> float:
        """Return the variance of the sequence.

        Returns:
            float: The variance.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 7, 8]).variance()
        9.7

        ```
        """
        return self.into(statistics.variance)
