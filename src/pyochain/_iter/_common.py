from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self

import cytoolz as cz
import more_itertools as mit

from .._core import CommonBase, Pipeable, SupportsRichComparison, get_config

if TYPE_CHECKING:
    from .._dict import Dict
    from .._results import Option
    from ._lazy import Iter, Seq, Vec


@dataclass(slots=True)
class Unzipped[T, V](Pipeable):
    left: Iter[T]
    right: Iter[V]


def convert_data[T](data: Iterable[T] | T, *more_data: T) -> Iterable[T]:
    return data if cz.itertoolz.isiterable(data) else (data, *more_data)


class CommonMethods[T](CommonBase[Iterable[T]]):
    _inner: Iterable[T]

    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_config().iter_repr(self._inner)})"

    def _eager[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], tuple[U, ...]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Seq[U]:
        from .._iter import Seq

        def _(data: Iterable[T]) -> Seq[U]:
            return Seq(factory(data, *args, **kwargs))

        return self.into(_)

    def _eager_mut[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], list[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Vec[U]:
        from .._iter import Vec

        def _(data: Iterable[T]) -> Vec[U]:
            return Vec(factory(data, *args, **kwargs))

        return self.into(_)

    def _lazy[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], Iterator[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[U]:
        from .._iter import Iter

        def _(data: Iterable[T]) -> Iter[U]:
            return Iter(factory(data, *args, **kwargs))

        return self.into(_)

    def eq(self, other: Self) -> bool:
        """Check if two Iterables are equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).eq(pc.Seq([1,2]))
        False
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2)))
        False
        >>> pc.Seq((1,2,3)).eq(pc.Vec([1,2,3]))
        True

        ```
        """
        return tuple(self._inner) == tuple(other._inner)

    def ne(self, other: Self) -> bool:
        """Check if two Iterables are not equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data are not equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2)))
        True
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) != tuple(other._inner)

    def le(self, other: Self) -> bool:
        """Check if this Iterable is less than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data of self is less than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).le(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).le(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) <= tuple(other._inner)

    def lt(self, other: Self) -> bool:
        """Check if this Iterable is less than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data of self is less than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).lt(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).lt(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) < tuple(other._inner)

    def gt(self, other: Self) -> bool:
        """Check if this Iterable is greater than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).gt(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).gt(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) > tuple(other._inner)

    def ge(self, other: Self) -> bool:
        """Check if this Iterable is greater than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Iter[T]|Seq[T]|Vec[T]` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).ge(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).ge(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) >= tuple(other._inner)

    def join(self: CommonMethods[str], sep: str) -> str:
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

    def unzip[U, V](self: CommonMethods[tuple[U, V]]) -> Unzipped[U, V]:
        """Converts an iterator of pairs into a pair of iterators.

        Returns:
            Unzipped[U, V]: dataclass with first and second iterators.

        `Iter.unzip()` consumes the iterator of pairs.

        Returns an Unzipped dataclass, containing two iterators:

        - one from the left elements of the pairs
        - one from the right elements.

        This function is, in some sense, the opposite of zip.
        ```python
        >>> import pyochain as pc
        >>> data = [(1, "a"), (2, "b"), (3, "c")]
        >>> unzipped = pc.Seq(data).unzip()
        >>> unzipped.left.collect()
        Seq(1, 2, 3)
        >>> unzipped.right.collect()
        Seq('a', 'b', 'c')

        ```
        """
        from ._lazy import Iter

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
        """Computes the index of the first element, without computing the previous combinations.

        The subsequences of iterable that are of length r can be ordered lexicographically.


        ValueError will be raised if the given element isn't one of the combinations of iterable.

        Equivalent to list(combinations(iterable, r)).index(element).

        Args:
            r (Iterable[T]): The combination to find the index of.

        Returns:
            int: The index of the combination.

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

    def nth(self, index: int) -> T:
        """Return the nth item at index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            T: The item at the specified index.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([10, 20]).nth(1)
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
        >>> models.nth(accuracy.argmax())
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
        >>> labels.nth(ages.argmin(key=cost))
        'bart'
        >>> # Age with fastest healing
        >>> ages.into(min, key=cost)
        10

        ```
        """
        return self.into(mit.argmin, key=key)

    def sum[U: int | float](self: CommonMethods[U]) -> U | Literal[0]:
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

    def min[U: int | float](self: CommonMethods[U]) -> U:
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

    def max[U: int | float](self: CommonMethods[U]) -> U:
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

    def all(self, predicate: Callable[[T], bool] = lambda x: bool(x)) -> bool:
        """Tests if every element of the iterator matches a predicate.

        `Iter.all()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if they all return true, then so does `Iter.all()`.

        If any of them return false, it returns false.

        An empty iterator returns true.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item. Defaults to checking truthiness.

        Returns:
            bool: True if all elements match the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, True]).all()
        True
        >>> pc.Seq([]).all()
        True
        >>> pc.Seq([1, 0]).all()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([2, 4, 6]).all(is_even)
        True

        ```
        """

        def _all(data: Iterable[T]) -> bool:
            return all(predicate(x) for x in data)

        return self.into(_all)

    def any(self, predicate: Callable[[T], bool] = lambda x: bool(x)) -> bool:
        """Tests if any element of the iterator matches a predicate.

        `Iter.any()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if any of them return true, then so does `Iter.any()`.

        If they all return false, it returns false.

        An empty iterator returns false.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item. Defaults to checking truthiness.

        Returns:
            bool: True if any element matches the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([0, 1]).any()
        True
        >>> pc.Seq(range(0)).any()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([1, 3, 4]).any(is_even)
        True

        ```
        """

        def _any(data: Iterable[T]) -> bool:
            return any(predicate(x) for x in data)

        return self.into(_any)

    def all_equal[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Return True if all items are equal.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all items are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 1, 1]).all_equal()
        True

        ```
        A function that accepts a single argument and returns a transformed version of each input item can be specified with key:
        ```python
        >>> pc.Seq("AaaA").all_equal(key=str.casefold)
        True
        >>> pc.Seq([1, 2, 3]).all_equal(key=lambda x: x < 10)
        True

        ```
        """
        return self.into(mit.all_equal, key=key)

    def all_unique[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Returns True if all the elements of iterable are unique.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all elements are unique, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq("ABCB").all_unique()
        False

        ```
        If a key function is specified, it will be used to make comparisons.
        ```python
        >>> pc.Seq("ABCb").all_unique()
        True
        >>> pc.Seq("ABCb").all_unique(str.lower)
        False

        ```
        The function returns as soon as the first non-unique element is encountered.

        Iterables with a mix of hashable and unhashable items can be used, but the function will be slower for unhashable items

        """
        return self.into(mit.all_unique, key=key)

    def is_sorted[U](
        self,
        key: Callable[[T], U] | None = None,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns True if the items of iterable are in sorted order.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.
            reverse (bool): Whether to check for descending order. Defaults to False.
            strict (bool): Whether to enforce strict sorting (no equal elements). Defaults to False.

        Returns:
            bool: True if items are sorted according to the criteria, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["1", "2", "3", "4", "5"]).is_sorted(key=int)
        True
        >>> pc.Seq([5, 4, 3, 1, 2]).is_sorted(reverse=True)
        False

        If strict, tests for strict sorting, that is, returns False if equal elements are found:
        ```python
        >>> pc.Seq([1, 2, 2]).is_sorted()
        True
        >>> pc.Seq([1, 2, 2]).is_sorted(strict=True)
        False

        ```

        The function returns False after encountering the first out-of-order item.

        This means it may produce results that differ from the built-in sorted function for objects with unusual comparison dynamics (like math.nan).

        If there are no out-of-order items, the iterable is exhausted.
        """
        return self.into(mit.is_sorted, key=key, reverse=reverse, strict=strict)

    def find(
        self,
        predicate: Callable[[T], bool],
    ) -> Option[T]:
        """Searches for an element of an iterator that satisfies a `predicate`.

        Takes a closure that returns true or false as `predicate`, and applies it to each element of the iterator.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Option[T]: The first element satisfying the predicate. `Some(value)` if found, `NONE` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def gt_five(x: int) -> bool:
        ...     return x > 5
        >>>
        >>> def gt_nine(x: int) -> bool:
        ...     return x > 9
        >>>
        >>> pc.Seq(range(10)).find(predicate=gt_five)
        Some(6)
        >>> pc.Seq(range(10)).find(predicate=gt_nine).unwrap_or("missing")
        'missing'

        ```
        """
        from .._results import Option

        def _find(data: Iterable[T]) -> Option[T]:
            return Option.from_(next(filter(predicate, data), None))

        return self.into(_find)

    def sort[U: SupportsRichComparison[Any]](
        self: CommonMethods[U],
        key: Callable[[U], Any] | None = None,
        *,
        reverse: bool = False,
    ) -> Vec[U]:
        """Sort the elements of the sequence.

        Note:
            This method must consume the entire iterable to perform the sort.
            The result is a new `Vec` over the sorted sequence.

        Args:
            key (Callable[[U], Any] | None): Function to extract a comparison key from each element. Defaults to None.
            reverse (bool): Whether to sort in descending order. Defaults to False.

        Returns:
            Vec[U]: A `Vec` with elements sorted.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).sort()
        Vec(1, 2, 3)

        ```
        """

        def _sort(data: Iterable[U]) -> list[U]:
            return sorted(data, reverse=reverse, key=key)

        return self._eager_mut(_sort)

    def tail(self, n: int) -> Seq[T]:
        """Return a tuple of the last n elements.

        Args:
            n (int): Number of elements to return.

        Returns:
            Seq[T]: A new Seq containing the last n elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).tail(2)
        Seq(2, 3)

        ```
        """
        return self._eager(functools.partial(cz.itertoolz.tail, n))

    def top_n(self, n: int, key: Callable[[T], Any] | None = None) -> Seq[T]:
        """Return a tuple of the top-n items according to key.

        Args:
            n (int): Number of top elements to return.
            key (Callable[[T], Any] | None): Function to extract a comparison key from each element. Defaults to None.

        Returns:
            Seq[T]: A new Seq containing the top-n elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 3, 2]).top_n(2)
        Seq(3, 2)

        ```
        """
        return self._eager(functools.partial(cz.itertoolz.topk, n, key=key))

    def union(self, *others: Iterable[T]) -> Seq[T]:
        """Return the union of this iterable and 'others'.

        Note:
            This method consumes inner data and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to include in the union.

        Returns:
            Seq[T]: A new Seq containing the union of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 2]).union([2, 3], [4]).iter().sort()
        Vec(1, 2, 3, 4)

        ```
        """

        def _union(data: Iterable[T]) -> tuple[T, ...]:
            return tuple(set(data).union(*others))

        return self._eager(_union)

    def intersection(self, *others: Iterable[T]) -> Seq[T]:
        """Return the elements common to this iterable and 'others'.

        Is the opposite of `difference`.

        See Also:
            - `difference`
            - `diff_symmetric`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to intersect with.

        Returns:
            Seq[T]: A new Seq containing the intersection of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 2]).intersection([2, 3], [2])
        Seq(2,)

        ```
        """

        def _intersection(data: Iterable[T]) -> tuple[T, ...]:
            return tuple(set(data).intersection(*others))

        return self._eager(_intersection)

    def difference(self, *others: Iterable[T]) -> Seq[T]:
        """Return the difference of this iterable and 'others'.

        See Also:
            - `intersection`
            - `diff_symmetric`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to subtract from this iterable.

        Returns:
            Seq[T]: A new Seq containing the difference of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 2]).difference([2, 3])
        Seq(1,)

        ```
        """

        def _difference(data: Iterable[T]) -> tuple[T, ...]:
            return tuple(set(data).difference(*others))

        return self._eager(_difference)

    def diff_symmetric(self, *others: Iterable[T]) -> Seq[T]:
        """Return the symmetric difference (XOR) of this iterable and 'others'.

        (Elements in either 'self' or 'others' but not in both).

        **See Also**:
            - `intersection`
            - `difference`

        Note:
            This method consumes inner data, unsorts it, and removes duplicates.

        Args:
            *others (Iterable[T]): Other iterables to compute the symmetric difference with.

        Returns:
            Seq[T]: A new Seq containing the symmetric difference of elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 2]).diff_symmetric([2, 3]).iter().sort()
        Vec(1, 3)
        >>> pc.Seq([1, 2, 3]).diff_symmetric([3, 4, 5]).iter().sort()
        Vec(1, 2, 4, 5)

        ```
        """

        def _symmetric_difference(data: Iterable[T]) -> tuple[T, ...]:
            return tuple(set(data).symmetric_difference(*others))

        return self._eager(_symmetric_difference)

    def most_common(self, n: int | None = None) -> Vec[tuple[T, int]]:
        """Return the n most common elements and their counts.

        If n is None, then all elements are returned.

        Args:
            n (int | None): Number of most common elements to return. Defaults to None (all elements).

        Returns:
            Vec[tuple[T, int]]: A new Seq containing tuples of (element, count).

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 1, 2, 3, 3, 3]).most_common(2)
        Vec((3, 3), (1, 2))

        ```
        """
        from collections import Counter

        def _most_common(data: Iterable[T]) -> list[tuple[T, int]]:
            return Counter(data).most_common(n)

        return self._eager_mut(_most_common)

    def rearrange[U: Sequence[Any]](
        self: CommonMethods[U], *indices: int
    ) -> Vec[list[U]]:
        """Rearrange elements in a given list of arrays by order indices.

        The last element (value) always remains in place.

        Args:
            *indices (int): indices specifying new order of keys in each array.

        Returns:
            Vec[list[U]]: A new Vec containing rearranged elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq([["A", "X", 1], ["A", "Y", 2], ["B", "X", 3], ["B", "Y", 4]])
        >>> data.rearrange(1, 0)
        Vec(['X', 'A', 1], ['Y', 'A', 2], ['X', 'B', 3], ['Y', 'B', 4])

        ```
        """

        def _check_bound(i: int, max_key_index: int) -> None:
            if i < 0 or i > max_key_index:
                msg = f"order index {i} out of range for row with {max_key_index + 1} keys"
                raise IndexError(
                    msg,
                )

        def _rearrange(in_arrs: Iterable[U]) -> list[list[U]]:
            order = indices
            out: list[list[U]] = []
            for arr in in_arrs:
                max_key_index: int = len(arr) - 2
                for i in order:
                    _check_bound(i, max_key_index)

                out.append([arr[i] for i in order] + [arr[-1]])

            return out

        return self._eager_mut(_rearrange)

    def with_keys[K](self, keys: Iterable[K]) -> Dict[K, T]:
        """Create a Dict by zipping the iterable with keys.

        Args:
            keys (Iterable[K]): Iterable of keys to pair with the values.

        Returns:
            Dict[K, T]: Dict with the provided keys and iterable values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> keys = ["a", "b", "c"]
        >>> values = [1, 2, 3]
        >>> pc.Seq(values).iter().with_keys(keys)
        {'a': 1, 'b': 2, 'c': 3}
        >>> # This is equivalent to:
        >>> pc.Iter(keys).zip(values).into(lambda x: pc.Dict(dict(x)))
        {'a': 1, 'b': 2, 'c': 3}

        ```
        """
        from .._dict import Dict

        def _with_keys(data: Iterable[T]) -> Dict[K, T]:
            return Dict(dict(zip(keys, data, strict=False)))

        return self.into(_with_keys)

    def with_values[V](self, values: Iterable[V]) -> Dict[T, V]:
        """Create a Dict by zipping the iterable with values.

        Args:
            values (Iterable[V]): Iterable of values to pair with the keys.

        Returns:
            Dict[T, V]: Dict with the iterable as keys and provided values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> keys = [1, 2, 3]
        >>> values = ["a", "b", "c"]
        >>> pc.Iter(keys).with_values(values)
        {1: 'a', 2: 'b', 3: 'c'}
        >>> # This is equivalent to:
        >>> pc.Iter(keys).zip(values).into(lambda x: pc.Dict(dict(x)))
        {1: 'a', 2: 'b', 3: 'c'}

        ```
        """
        from .._dict import Dict

        def _with_values(data: Iterable[T]) -> Dict[T, V]:
            return Dict(dict(zip(data, values, strict=False)))

        return self.into(_with_values)

    def reduce_by[K](
        self,
        key: Callable[[T], K],
        binop: Callable[[T, T], T],
    ) -> Dict[K, T]:
        """Perform a simultaneous groupby and reduction.

        Args:
            key (Callable[[T], K]): Function to compute the key for grouping.
            binop (Callable[[T, T], T]): Binary operation to reduce the grouped elements.

        Returns:
            Dict[K, T]: Dict with grouped and reduced elements.

        Example:
        ```python
        >>> from collections.abc import Iterable
        >>> import pyochain as pc
        >>> from operator import add, mul
        >>>
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>>
        >>> def group_reduce(data: Iterable[int]) -> int:
        ...     return pc.Iter(data).reduce(add)
        >>>
        >>> data = pc.Seq([1, 2, 3, 4, 5])
        >>> data.iter().reduce_by(is_even, add)
        {False: 9, True: 6}
        >>> data.iter().group_by(is_even).map_values(group_reduce)
        {False: 9, True: 6}

        ```
        But the former does not build the intermediate groups, allowing it to operate in much less space.

        This makes it suitable for larger datasets that do not fit comfortably in memory

        Simple Examples:
        ```python
        >>> pc.Iter([1, 2, 3, 4, 5]).reduce_by(is_even, add)
        {False: 9, True: 6}
        >>> pc.Iter([1, 2, 3, 4, 5]).reduce_by(is_even, mul)
        {False: 15, True: 8}

        ```
        """
        from .._dict import Dict

        def _reduce_by(data: Iterable[T]) -> Dict[K, T]:
            return Dict(cz.itertoolz.reduceby(key, binop, data))

        return self.into(_reduce_by)

    def group_by[K](self, on: Callable[[T], K]) -> Dict[K, list[T]]:
        """Group elements by key function and return a Dict result.

        Args:
            on (Callable[[T], K]): Function to compute the key for grouping.

        Returns:
            Dict[K, list[T]]: Dict with grouped elements as lists.

        Example:
        ```python
        >>> import pyochain as pc
        >>> names = [
        ...     "Alice",
        ...     "Bob",
        ...     "Charlie",
        ...     "Dan",
        ...     "Edith",
        ...     "Frank",
        ... ]
        >>> pc.Iter(names).group_by(len).sort()
        ... # doctest: +NORMALIZE_WHITESPACE
        {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
        >>>
        >>> iseven = lambda x: x % 2 == 0
        >>> pc.Iter([1, 2, 3, 4, 5, 6, 7, 8]).group_by(iseven)
        ... # doctest: +NORMALIZE_WHITESPACE
        {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

        ```
        Non-callable keys imply grouping on a member.
        ```python
        >>> data = [
        ...     {"name": "Alice", "gender": "F"},
        ...     {"name": "Bob", "gender": "M"},
        ...     {"name": "Charlie", "gender": "M"},
        ... ]
        >>> pc.Iter(data).group_by("gender").sort()
        ... # doctest: +NORMALIZE_WHITESPACE
        {'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'}, {'name': 'Charlie', 'gender': 'M'}]}

        ```
        """
        from .._dict import Dict

        def _group_by(data: Iterable[T]) -> Dict[K, list[T]]:
            return Dict(cz.itertoolz.groupby(on, data))

        return self.into(_group_by)

    def frequencies(self) -> Dict[T, int]:
        """Find number of occurrences of each value in the iterable.

        Returns:
            Dict[T, int]: Dict with element frequencies as counts.

        ```python
        >>> import pyochain as pc
        >>> data = ["cat", "cat", "ox", "pig", "pig", "cat"]
        >>> pc.Iter(data).frequencies()
        {'cat': 3, 'ox': 1, 'pig': 2}

        ```
        """
        from .._dict import Dict

        def _frequencies(data: Iterable[T]) -> Dict[T, int]:
            return Dict(cz.itertoolz.frequencies(data))

        return self.into(_frequencies)

    def count_by[K](self, key: Callable[[T], K]) -> Dict[K, int]:
        """Count elements of a collection by a key function.

        Args:
            key (Callable[[T], K]): Function to compute the key for counting.

        Returns:
            Dict[K, int]: Dict with count of elements for each key.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(["cat", "mouse", "dog"]).count_by(len)
        {3: 2, 5: 1}
        >>> def iseven(x):
        ...     return x % 2 == 0
        >>> pc.Iter([1, 2, 3]).count_by(iseven)
        {False: 2, True: 1}

        ```
        """
        from .._dict import Dict

        def _count_by(data: Iterable[T]) -> Dict[K, int]:
            return Dict(cz.recipes.countby(key, data))

        return self.into(_count_by)

    def to_records[U: Sequence[Any]](self: CommonMethods[U]) -> Dict[Any, Any]:
        """Transform an iterable of nested sequences into a nested dictionary.

        - Each inner sequence represents a path to a value in the dictionary.
        - The last element of each sequence is treated as the value
        - All preceding elements are treated as keys leading to that value.

        Returns:
            Dict[Any, Any]: Nested dictionary constructed from the sequences.

        Example:
        ```python
        >>> import pyochain as pc
        >>> arrays = [["a", "b", 1], ["a", "c", 2], ["d", 3]]
        >>> pc.Seq(arrays).to_records()
        {'a': {'b': 1, 'c': 2}, 'd': 3}

        ```
        """
        from .._dict import Dict

        def _from_nested(
            arrays: Iterable[Sequence[Any]],
            parent: dict[Any, Any] | None = None,
        ) -> dict[Any, Any]:
            d: dict[Any, Any] = parent or {}
            for arr in arrays:
                if len(arr) > 1:
                    head, *tail = arr
                    if len(tail) == 1:
                        d[head] = tail[0]
                    else:
                        d[head] = _from_nested([tail], d.get(head, {}))
            return d

        return Dict(self.into(_from_nested))
