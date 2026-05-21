from __future__ import annotations

import functools
import itertools
from abc import ABC
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Concatenate, overload

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from .._types import SupportsComparison, SupportsRichComparison
from ..rs import NONE, Option, Result, Some, option
from ._iterable import PyoIterable

if TYPE_CHECKING:
    from .._seq import Seq
    from .._vec import Vec
    from ..rs import Option, Result
    from ._sequences import PyoMutableSequence


class PyoIterator[T](PyoIterable[T], Iterator[T], ABC):
    """Extends `PyoIterable[T]` and `collections.abc.Iterator[T]`.

    Is the base class for `Iter[T]`.

    All concrete subclasses must implement the required `Iterator` dunder methods:

    - `__iter__`
    - `__next__`

    Example:
        ```python
        >>> from pyochain.abc import PyoIterator
        >>> class Count(PyoIterator[int]):
        ...     def __init__(self, start: int = 0):
        ...         self.current = start
        ...
        ...     def __iter__(self):
        ...         return self
        ...
        ...     def __next__(self):
        ...         val = self.current
        ...         self.current += 1
        ...         return val
        >>>
        >>> counter = Count(5)
        >>> counter.next()
        Some(5)
        >>> counter.next()
        Some(6)
        >>> counter.iter().take(3).collect()
        Seq(7, 8, 9)

        ```
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def nth(self, n: int) -> Option[T]:
        """Return the nth item of the `Iterable` at the specified *n*.

        This is similar to `__getitem__` but for lazy `Iterators`.

        If *n* is out of bounds, returns `NONE`.

        Args:
            n (int): The index of the item to retrieve.

        Returns:
            Option[T]: `Some(item)` at the specified *n*.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter([10, 20]).nth(1)
            Some(20)
            >>> Iter([10, 20]).nth(3)
            NONE

            ```
        """
        try:
            return Some(next(itertools.islice(iter(self), n, n + 1)))
        except StopIteration:
            return NONE

    def eq(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** and *other* contain the same items in the same order.

        Comparison is performed element by element.

        Two `Iterable`s are equal only if:

        - every compared pair of elements is equal
        - and both iterables are exhausted at the same time

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` when both iterables yield the same sequence of values.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).eq(Seq((1, 2, 3)))
            True
            >>> Iter((1, 2, 3)).eq((1, 2, 4))
            False
            >>> Iter((1, 2, 3)).eq((1, 2))
            False
            >>> Iter((1, 2)).eq((1, 2, 3))
            False

            ```
        """
        return tls.eq(iter(self), other)

    def ne(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** and *other* differ in value or length.

        This is the logical opposite of `eq()`.

        The result becomes `True` as soon as:

        - a pair of compared elements is not equal
        - or one iterable ends before the other

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` when the two iterables are not equal.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).ne(Seq((1, 2, 3)))
            False
            >>> Iter((1, 2, 3)).ne((1, 2, 4))
            True
            >>> Iter((1, 2, 3)).ne((1, 2))
            True

            ```
        """
        return tls.ne(iter(self), other)

    def le(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically less than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the shorter iterable is considered smaller.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** is smaller than *other*, or equal to it.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).le((1, 2, 3))
            True
            >>> Iter((1, 2, 3)).le((1, 2, 3))
            True
            >>> Iter((1, 3)).le((1, 2, 9))
            False

            ```
        """
        return tls.le(iter(self), other)

    def lt(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically strictly less than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, a shorter iterable is strictly smaller than a longer one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** compares strictly before *other*.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).lt((1, 2, 3))
            True
            >>> Iter((1, 2, 3)).lt((1, 2, 3))
            False
            >>> Iter((1, 2, 3)).lt((1, 3))
            True

            ```
        """
        return tls.lt(iter(self), other)

    def gt(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically strictly greater than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, the longer iterable is strictly greater than the shorter one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** compares strictly after *other*.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).gt((1, 2))
            True
            >>> Iter((1, 3)).gt((1, 2, 9))
            True
            >>> Iter((1, 2)).gt((1, 2, 3))
            False

            ```
        """
        return tls.gt(iter(self), other)

    def ge(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically greater than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the longer iterable is considered
        greater.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an iterator.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** is greater than *other*, or equal to it.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).ge((1, 2))
            True
            >>> Iter((1, 2, 3)).ge((1, 2, 3))
            True
            >>> Iter((1, 2)).ge((1, 2, 3))
            False

            ```
        """
        return tls.ge(iter(self), other)

    def next(self) -> Option[T]:
        """Return the next element in the `Iterator`.

        The actual `__next__()` method must be conform to the Python `Iterator` Protocol, and is what will be actually called if you iterate over the `PyoIterator` instance.

        `PyoIterator::next` is a convenience method that wraps the result in an `Option` to handle exhaustion gracefully, for custom use cases.

        Returns:
            Option[T]: The next element in the iterator. `Some[T]`, or `NONE` if the iterator is exhausted.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> it = Seq((1, 2, 3)).iter()
            >>> it.next().unwrap()
            1
            >>> it.next().unwrap()
            2

            ```
        """
        return option(next(self, None))

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Apply a function of two arguments cumulatively to the items of an iterable, from left to right.

        This effectively reduces the `Iterator` to a single value.

        If initial is present, it is placed before the items of the `Iterator` in the calculation.

        It then serves as a default when the `Iterator` is empty.

        Args:
            func (Callable[[T, T], T]): Function to apply cumulatively to the items of the iterable.

        Returns:
            T: Single value resulting from cumulative reduction.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).reduce(lambda a, b: a + b)
            6

            ```
        """
        return functools.reduce(func, self)

    def fold[B](self, init: B, func: Callable[[B, T], B]) -> B:
        """Fold every element of the `Iterator` into an accumulator by applying an operation, returning the final result.

        Args:
            init (B): Initial value for the accumulator.
            func (Callable[[B, T], B]): Function that takes the accumulator and current element,
                returning the new accumulator value.

        Returns:
            B: The final accumulated value.

        Note:
            This is similar to `reduce()` but with an initial value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = (1, 2, 3)
            >>> Iter(data).fold(0, lambda acc, x: acc + x)
            6
            >>> Iter(data).fold(10, lambda acc, x: acc + x)
            16
            >>> Iter(("a", "b", "c")).fold("", lambda acc, x: acc + x)
            'abc'

            ```
        """
        return functools.reduce(func, self, init)

    @overload
    def fold_star[**P, B](
        self: PyoIterator[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        init: B,
        func: Callable[[Any], B],  # pyright: ignore[reportExplicitAny]
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, **P, B](
        self: PyoIterator[tuple[T1, T2]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, **P, B](
        self: PyoIterator[tuple[T1, T2, T3]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, T9, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    def fold_star[U: Iterable[Any], **P, B](
        self: PyoIterator[U],
        init: B,
        func: Callable[..., B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B:
        """Fold every element of the `Iterator` into an accumulator by applying an operation, returning the final result.

        Use this when the items of the `Iterator` are themselves iterables (e.g., tuples), and you want to unpack them as arguments to the folding function.

        Args:
            init (B): Initial value for the accumulator.
            func (Callable[..., B]): Function that takes the accumulator and current element, returning the new accumulator value.
            *args (P.args): Additional positional arguments to pass to **func**.
            **kwargs (P.kwargs): Additional keyword arguments to pass to **func**.

        Returns:
            B: The final accumulated value.

        Note:
            This is similar to `Iter::reduce` but with an initial value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = ((1, 2), (3, 4))
            >>> Iter(data).fold_star(0, lambda acc, x, y: acc + x + y)
            10
            >>> data = (("a", "b"), ("c", "d"))
            >>> Iter(data).fold_star("", lambda acc, x, y: acc + x + y)
            'abcd'

            ```
        """

        def _reducer(acc: B, item: U) -> B:
            return func(acc, *item, *args, **kwargs)

        return functools.reduce(_reducer, self, init)

    def find(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Searches for an element of an iterator that satisfies a `predicate`.

        Takes a closure that returns true or false as `predicate`, and applies it to each element of the iterator.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Option[T]: The first element satisfying the predicate. `Some(value)` if found, `NONE` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter, Range
            >>> def gt_five(x: int) -> bool:
            ...     return x > 5
            >>>
            >>> def gt_nine(x: int) -> bool:
            ...     return x > 9
            >>> data = Range(0, 10)
            >>> data.iter().find(predicate=gt_five)
            Some(6)
            >>> data.iter().find(predicate=gt_nine).unwrap_or("missing")
            'missing'

            ```
        """
        return option(next(filter(predicate, self), None))

    def try_find[E](
        self, predicate: Callable[[T], Result[bool, E]]
    ) -> Result[Option[T], E]:
        """Applies a function returning `Result[bool, E]` to find first matching element.

        Short-circuits: stops at the first successful `True` or on the first error.

        Args:
            predicate (Callable[[T], Result[bool, E]]): Function returning a `Result[bool, E]`.

        Returns:
            Result[Option[T], E]: The first matching element, or the first error.

        Example:
            ```python
            >>> from pyochain import Ok, Result, Err, Range
            >>> def is_even(x: int) -> Result[bool, str]:
            ...     return Ok(x % 2 == 0) if x >= 0 else Err("negative number")
            >>>
            >>> Range(1, 6).iter().try_find(is_even)
            Ok(Some(2))

            ```
        """
        return tls.try_find(iter(self), predicate)

    def try_fold[B, E](
        self, init: B, func: Callable[[B, T], Result[B, E]]
    ) -> Result[B, E]:
        """Folds every element into an accumulator, short-circuiting on error.

        Applies **func** cumulatively to items and the accumulator.

        If **func** returns an error, stops and returns that error.

        Args:
            init (B): Initial accumulator value.
            func (Callable[[B, T], Result[B, E]]): Function that takes the accumulator and element, returns a `Result[B, E]`.

        Returns:
            Result[B, E]: Final accumulator or the first error.

        Example:
            ```python
            >>> from pyochain import Iter, Ok, Err, Result
            >>> def checked_add(acc: int, x: int) -> Result[int, str]:
            ...     new_val = acc + x
            ...     if new_val > 100:
            ...         return Err("overflow")
            ...     return Ok(new_val)
            >>>
            >>> Iter((1, 2, 3)).try_fold(0, checked_add)
            Ok(6)
            >>> Iter([50, 40, 20]).try_fold(0, checked_add)
            Err('overflow')
            >>> Iter(()).try_fold(0, checked_add)
            Ok(0)

            ```
        """
        return tls.try_fold(iter(self), init, func)

    def try_reduce[E](
        self, func: Callable[[T, T], Result[T, E]]
    ) -> Result[Option[T], E]:
        """Reduces elements to a single one, short-circuiting on error.

        Uses the first element as the initial accumulator. If **func** returns an error, stops immediately.

        Args:
            func (Callable[[T, T], Result[T, E]]): Function that reduces two items, returns a `Result[T, E]`.

        Returns:
            Result[Option[T], E]: Final accumulated value or the first error. Returns `Ok(NONE)` for empty iterable.

        Example:
            ```python
            >>> from pyochain import Iter, Ok, Err, Result
            >>> def checked_add(x: int, y: int) -> Result[int, str]:
            ...     if x + y > 100:
            ...         return Err("overflow")
            ...     return Ok(x + y)
            >>>
            >>> Iter((1, 2, 3)).try_reduce(checked_add)
            Ok(Some(6))
            >>> Iter([50, 60]).try_reduce(checked_add)
            Err('overflow')
            >>> Iter(()).try_reduce(checked_add)
            Ok(NONE)

            ```
        """
        return tls.try_reduce(iter(self), func)

    def is_sorted[U: SupportsComparison[Any]](
        self: PyoIterator[U], *, reverse: bool = False, strict: bool = False
    ) -> bool:
        """Returns `True` if the items of the `Iterator` are in sorted order.

        The elements of the `Iterator` must support comparison operations.

        The function returns `False` after encountering the first out-of-order item.

        If there are no out-of-order items, the `Iterator` is exhausted.

        Credits to **more-itertools** for the implementation.

        See Also:
            - `is_sorted_by()`: If your elements do not support comparison operations directly, or you want to sort based on a specific attribute or transformation.

        Args:
            reverse (bool): Whether to check for descending order.
            strict (bool): Whether to enforce strict sorting (no equal elements).

        Returns:
            bool: `True` if items are sorted according to the criteria, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3, 4, 5)).is_sorted()
            True

            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            >>> Iter([1, 2, 2]).is_sorted()
            True
            >>> Iter([1, 2, 2]).is_sorted(strict=True)
            False

            ```
        """
        return tls.is_sorted(iter(self), reverse=reverse, strict=strict)

    def is_sorted_by(
        self,
        key: Callable[[T], SupportsComparison[Any]],  # pyright: ignore[reportExplicitAny]
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns `True` if the items of the `Iterator` are in sorted order according to the key function.

        The function returns `False` after encountering the first out-of-order item.

        If there are no out-of-order items, the `Iterator` is exhausted.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], SupportsComparison[Any]]): Function to extract a comparison key from each element.
            reverse (bool): Whether to check for descending order.
            strict (bool): Whether to enforce strict sorting (no equal elements).

        Returns:
            bool: `True` if items are sorted according to the criteria, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(["1", "2", "3", "4", "5"]).is_sorted_by(int)
            True
            >>> Iter(["5", "4", "3", "1", "2"]).is_sorted_by(int, reverse=True)
            False

            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            >>> Iter(["1", "2", "2"]).is_sorted_by(int)
            True
            >>> Iter(["1", "2", "2"]).is_sorted_by(key=int, strict=True)
            False

            ```
        """
        return tls.is_sorted_by(iter(self), key, reverse=reverse, strict=strict)

    def all_equal[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Return `True` if all items of the `Iterator` are equal.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison.

        Returns:
            bool: `True` if all items are equal, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter("AaaA").all_equal(key=str.casefold)
            True
            >>> Iter((1, 2, 3)).all_equal(key=lambda x: x < 10)
            True

            ```
        """
        iterator = itertools.groupby(iter(self), key)
        for _first in iterator:
            for _second in iterator:
                return False
            return True
        return True

    def argmax[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator`.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter("abcdefghabcd").argmax()
            7
            >>> Iter([0, 1, 2, 3, 3, 2, 1, 0]).argmax()
            3

            ```
            For example, identify the best machine learning model:
            ```python
            >>> models = Seq(["svm", "random forest", "knn", "naïve bayes"])
            >>> accuracy = Seq([68, 61, 84, 72])
            >>> # Most accurate model
            >>> models.get(accuracy.iter().argmax()).unwrap()
            'knn'
            >>>
            >>> # Best accuracy
            >>> accuracy.max()
            84

            ```
        """
        it = iter(self)
        if key is not None:
            it = map(key, it)
        return max(enumerate(it), key=itemgetter(1))[0]

    def argmin[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator`.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the minimum value.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> # Example 1: Basic usage
            >>> Iter("efghabcdijkl").argmin()
            4
            >>> Iter((3, 2, 1, 0, 4, 2, 1, 0)).argmin()
            3
            >>> # Example 2: look up a label corresponding to the position of a value that minimizes a cost function
            >>> def cost(x: int) -> float:
            ...     "Days for a wound to heal given a subject's age."
            ...     return x**2 - 20 * x + 150
            >>>
            >>> labels = Seq(("homer", "marge", "bart", "lisa", "maggie"))
            >>> ages = Seq((35, 30, 10, 9, 1))
            >>> # Fastest healing family member
            >>> labels.get(ages.iter().argmin(key=cost)).unwrap()
            'bart'
            >>> # Age with fastest healing
            >>> ages.min_by(key=cost)
            10

            ```
        """
        it = iter(self)
        if key is not None:
            it = map(key, it)
        return min(enumerate(it), key=itemgetter(1))[0]

    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],  # pyright: ignore[reportExplicitAny]
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the `Iterator` by applying a function to each element in the `Iterable`.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Args:
            func (Callable[Concatenate[T, P], Any]): Function to apply to each element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).for_each(lambda x: print(x + 1))
            2
            3
            4

            ```
        """
        tls.for_each(iter(self), func, *args, **kwargs)

    @overload
    def for_each_star[T1, T2, **P, R](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[Concatenate[T1, T2, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, **P, R](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[Concatenate[T1, T2, T3, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        func: Callable[Concatenate[T1, T2, T3, T4, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, T9, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    def for_each_star[U: tuple[Any, ...], **P, R](
        self: PyoIterator[U],
        func: Callable[..., R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the `Iterator` by applying a function to each unpacked item in the `Iterable` element.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Each item yielded by the `Iterator` is expected to be an `Iterable` itself (e.g., a tuple or list),
        and its elements are unpacked as arguments to the provided function.

        This is often used after methods like `zip()` or `enumerate()` that yield tuples.

        Args:
            func (Callable[..., R]): Function to apply to each unpacked element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(((1, 2), (3, 4))).for_each_star(lambda x, y: print(x + y))
            3
            7

            ```
        """
        tls.for_each_star(iter(self), func, *args, **kwargs)

    def try_for_each[E](self, f: Callable[[T], Result[Any, E]]) -> Result[tuple[()], E]:  # pyright: ignore[reportExplicitAny]
        """Applies a fallible function to each item in the `Iterator`, stopping at the first error and returning that error.

        This can also be thought of as the fallible form of `.for_each()`.

        Args:
            f (Callable[[T], Result[Any, E]]): A function that takes an item of type `T` and returns a `Result`.

        Returns:
            Result[tuple[()], E]: Returns `Ok(())` if all applications of **f** were successful (i.e., returned `Ok`), or the first error `E` encountered.

        Example:
            ```python
            >>> from pyochain import Iter, Result, Ok, Err
            >>> def validate_positive(n: int) -> Result[tuple[()], str]:
            ...     if n > 0:
            ...         return Ok("success")
            ...     return Err(f"Value {n} is not positive")
            >>>
            >>> Iter((1, 2, 3, 4, 5)).try_for_each(validate_positive)
            Ok(())
            >>> # Short-circuit on first error:
            >>> Iter((1, 2, -1, 4)).try_for_each(validate_positive)
            Err('Value -1 is not positive')

            ```
        """
        return tls.try_for_each(iter(self), f)

    @overload
    def collect_into(self, collection: Vec[T]) -> Vec[T]: ...
    @overload
    def collect_into(
        self, collection: PyoMutableSequence[T]
    ) -> PyoMutableSequence[T]: ...
    @overload
    def collect_into(self, collection: list[T]) -> list[T]: ...
    def collect_into(self, collection: MutableSequence[T]) -> MutableSequence[T]:
        """Collects all the items from the `Iterator` into a `MutableSequence`.

        The `MutableSequence` is then returned, so the call chain can be continued.

        This is useful when you already have a `MutableSequence` and want to add the `Iterator` items to it.

        This method is a convenience method to call `MutableSequence.extend()`, but instead of being called on a `MutableSequence`, it's called on an `Iterator`.

        Args:
            collection (MutableSequence[T]): A mutable collection to collect items into.

        Returns:
            MutableSequence[T]: The same mutable collection passed as argument, now containing the collected items.

        Example:
            Basic usage:
            ```python
            >>> from pyochain import Seq, Iter, Vec
            >>> a = Seq((1, 2, 3))
            >>> vec = Vec.from_ref([0, 1])
            >>> a.iter().map(lambda x: x * 2).collect_into(vec)
            Vec(0, 1, 2, 4, 6)
            >>> a.iter().map(lambda x: x * 10).collect_into(vec)
            Vec(0, 1, 2, 4, 6, 10, 20, 30)

            ```
            The returned mutable sequence can be used to continue the call chain:
            ```python
            >>> from pyochain import Seq, Vec
            >>> a = Seq((1, 2, 3))
            >>> vec = Vec(())
            >>> a.iter().collect_into(vec).length() == vec.length()
            True
            >>> a.iter().collect_into(vec).length() == vec.length()
            True

            ```
        """
        collection.extend(iter(self))
        return collection

    @overload
    def try_collect[U](self: PyoIterator[Option[U]]) -> Option[Vec[U]]: ...
    @overload
    def try_collect[U, E](self: PyoIterator[Result[U, E]]) -> Option[Vec[U]]: ...
    def try_collect[U](
        self: PyoIterator[Option[U]] | PyoIterator[Result[U, Any]],  # pyright: ignore[reportExplicitAny]
    ) -> Option[Vec[U]]:
        """Fallibly transforms **self** into a `Vec`, short circuiting if a failure is encountered.

        `try_collect()` is a variation of `collect()` that allows fallible conversions during collection.

        Its main use case is simplifying conversions from iterators yielding `Option[T]` or `Result[T, E]` into `Option[Vec[T]]`.

        Also, if a failure is encountered during `try_collect()`, the `Iter` is still valid and may continue to be used, in which case it will continue iterating starting after the element that triggered the failure.

        See the last example below for an example of how this works.

        Note:
            This method return `Vec[U]` instead of being customizable, because the underlying data structure must be mutable in order to build up the collection.

        Returns:
            Option[Vec[U]]: `Some[Vec[U]]` if all elements were successfully collected, or `NONE` if a failure was encountered.

        Example:
            ```python
            >>> from pyochain import Iter, Some, Ok, Err, NONE, Vec
            >>> # Successfully collecting an iterator of Option[int] into Option[Vec[int]]:
            >>> Iter((Some(1), Some(2), Some(3))).try_collect()
            Some(Vec(1, 2, 3))
            >>> # Failing to collect in the same way:
            >>> Iter((Some(1), Some(2), NONE, Some(3))).try_collect()
            NONE
            >>> # A similar example, but with Result:
            >>> Iter((Ok(1), Ok(2), Ok(3))).try_collect()
            Some(Vec(1, 2, 3))
            >>> Iter((Ok(1), Err("error"), Ok(3))).try_collect()
            NONE
            >>> def external_fn(x: int) -> Option[int]:
            ...     if x % 2 == 0:
            ...         return Some(x)
            ...     return NONE
            >>>
            >>> Iter((1, 2, 3, 4)).map(external_fn).try_collect()
            NONE
            >>> # Demonstrating that the iterator remains usable after a failure:
            >>> it = Iter((Some(1), NONE, Some(3), Some(4)))
            >>> it.try_collect()
            NONE
            >>> it.try_collect()
            Some(Vec(3, 4))

            ```
        """
        from .._vec import Vec

        return tls.try_collect(iter(self)).map(Vec.from_ref)

    def sort[U: SupportsRichComparison[Any]](
        self: PyoIterator[U], *, reverse: bool = False
    ) -> Vec[U]:
        """Sort the elements of the `Iterator`.

        The elements must support rich comparison operations (i.e., they must implement the necessary comparison dunder methods).

        Note:
            This method must consume the entire `Iterator` to perform the sort.

            The result is a new `Vec` over the sorted sequence.

        Args:
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[U]: A `Vec` with elements sorted.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((3, 1, 2)).sort()
            Vec(1, 2, 3)

            ```
        """
        from .._vec import Vec

        return Vec.from_ref(sorted(iter(self), reverse=reverse))

    def sort_by(
        self,
        key: Callable[[T], SupportsRichComparison[Any]],  # pyright: ignore[reportExplicitAny]
        *,
        reverse: bool = False,
    ) -> Vec[T]:
        """Sort the elements of the sequence transformed by the key function.

        Note:
            This method must consume the entire `Iterator` to perform the sort.

            The result is a new `Vec` over the sorted sequence.

        Args:
            key (Callable[[T], SupportsRichComparison[Any]]): Function to extract a comparison key from each element.
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[T]: A `Vec` with elements sorted.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> str_numbers = ("3", "1", "2")
            >>> Iter(str_numbers).sort_by(int)
            Vec('1', '2', '3')
            >>> Iter(str_numbers).sort_by(int, reverse=True)
            Vec('3', '2', '1')
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            >>>
            >>> peoples = (
            ...     Person("Alice", 30),
            ...     Person("Bob", 25),
            ...     Person("Charlie", 35),
            ... )
            >>> sorted_names = (
            ...     Iter(peoples)
            ...     .sort_by(lambda x: x.age)
            ...     .iter()
            ...     .map(lambda x: x.name)
            ...     .collect()
            ... )
            >>> sorted_names
            Seq('Bob', 'Alice', 'Charlie')

            ```
        """
        from .._vec import Vec

        return Vec.from_ref(sorted(iter(self), reverse=reverse, key=key))

    def tail(self, n: int) -> Seq[T]:
        """Return a `Seq` of the last **n** elements of the `Iterator`.

        Args:
            n (int): Number of elements to return.

        Returns:
            Seq[T]: A `Seq` containing the last **n** elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).tail(2)
            Seq(2, 3)

            ```
        """
        from collections import deque

        from .._seq import Seq

        # TODO: we should move this to Rust and make it fully lazy.
        # Here we recollect it in a Seq to clearly indicate that we need to consume the entire iterator to get the tail.
        # Alternatively, add `deque` wrapper to public API, and `from_ref` it here.
        return Seq(deque(iter(self), n))
