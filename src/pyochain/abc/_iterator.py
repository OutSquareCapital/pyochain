from __future__ import annotations

import functools
import itertools
from abc import ABC
from collections.abc import Iterator
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Concatenate, overload

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from ..rs import NONE, Option, Result, Some, option
from ._iterable import PyoIterable

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, MutableSequence

    from .._types import SupportsComparison, SupportsRichComparison
    from .._vec import Vec
    from ..collections import Deque
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

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def count(self) -> int:
        """Consume the `Iterator` and return the number of elements it contained.

        Returns:
            int: The count of elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = Iter((1, 2, 3))
            >>> data.count()
            3
            >>> # data is now empty
            >>> data.count()
            0

            ```
        """
        return tls.length(iter(self))

    def all(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if every element of the `Iterator` is truthy.

        `PyoIterator::.all` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterator`, and if they all return true, then so does `PyoIterator::.all`.

        If any of them return false, it returns false.

        An empty `Iterator` returns true.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if all elements match the predicate, False otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, True)).all()
            True
            >>> Iter(()).all()
            True
            >>> Iter((1, 0)).all()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>>
            >>> Iter((2, 4, 6)).all(is_even)
            True
            >>> Iter(("a", "", "c")).all()
            False
            >>> Iter((1, None, 3)).all()
            False

            ```
        """
        if predicate is None:
            return all(iter(self))
        return all(predicate(x) for x in iter(self))

    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if any element of the `Iterator` is truthy.

        `PyoIterator::.any` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterator`, and if any of them return true, then so does `PyoIterator::.any`.

        If they all return false, it returns false.

        An empty `Iterator` returns false.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if any element matches the predicate, False otherwise.

        Example:
            ```python
            >>> from pyochain import Iter, Range
            >>> Iter((0, 1)).any()
            True
            >>> Range(0, 0).iter().any()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>> Iter((1, 3, 4)).any(is_even)
            True

            ```
        """
        if predicate is None:
            return any(iter(self))
        return any(predicate(x) for x in iter(self))

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
            [`PyoIterator::is_sorted_by`][is_sorted_by] if your elements do not support comparison operations directly, or you want to sort based on a specific attribute or transformation.

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

    def arg_max(self) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator`.

        Credits to more-itertools for the implementation.

        Returns:
            int: The index of the maximum value.

        Example:
            Basic usage:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter("abcdefghabcd").arg_max()
            7
            >>> Iter((0, 1, 2, 3, 3, 2, 1, 0)).arg_max()
            3

            ```
            Identify the best machine learning model:
            ```python
            >>> models = Seq(("svm", "random forest", "knn", "naïve bayes"))
            >>> accuracy = Seq((68, 61, 84, 72))
            >>> # Most accurate model
            >>> models.get(accuracy.iter().arg_max()).unwrap()
            'knn'
            >>> # Best accuracy
            >>> accuracy.iter().max()
            84

            ```
        """
        return max(enumerate(iter(self)), key=itemgetter(1))[0]

    def arg_max_by[U](self, key: Callable[[T], U]) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator` based on a *key* function.

        The *key* function must accept a single argument and return a transformed, comparable version of each input item.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U]): Function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        Example:
            Basic usage:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter(("a", "bbb", "cc")).arg_max_by(len)
            1
            >>> Iter(("Alice", "bob", "charlie")).arg_max_by(str.lower)
            2

            ```
            Identify the best machine learning model:
            ```python
            >>> models = Seq(("svm", "random forest", "knn", "naïve bayes"))
            >>> accuracy = Seq(("68", "61", "84", "72"))
            >>> # Most accurate model
            >>> models.get(accuracy.iter().arg_max_by(int)).unwrap()
            'knn'
            >>> # Best accuracy
            >>> accuracy.iter().max_by(int)
            '84'

            ```
        """
        return max(enumerate(map(key, iter(self))), key=itemgetter(1))[0]

    def arg_min(self) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator`.

        Credits to more-itertools for the implementation.

        Returns:
            int: The index of the minimum value.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> # Example 1: Basic usage
            >>> Iter("efghabcdijkl").arg_min()
            4
            >>> Iter((3, 2, 1, 0, 4, 2, 1, 0)).arg_min()
            3

            ```
        """
        return min(enumerate(iter(self)), key=itemgetter(1))[0]

    def arg_min_by[U](self, key: Callable[[T], U]) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator` based on a *key* function.

        The *key* function must accept a single argument and return a transformed, comparable version of each input item.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U]): Function to determine the value for comparison.

        Returns:
            int: The index of the minimum value.

        Example:
            Basic usage:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter(("aaa", "b", "cc")).arg_min_by(len)
            1
            >>> Iter(("Alice", "bob", "Charlie")).arg_min_by(str.lower)
            0

            ```
            Identify the best machine learning model:
            ```python
            >>> def cost(x: int) -> float:
            ...     "Days for a wound to heal given a subject's age."
            ...     return x**2 - 20 * x + 150
            >>>
            >>> labels = Seq(("homer", "marge", "bart", "lisa", "maggie"))
            >>> ages = Seq((35, 30, 10, 9, 1))
            >>> # Fastest healing family member
            >>> labels.get(ages.iter().arg_min_by(cost)).unwrap()
            'bart'
            >>> # Age with fastest healing
            >>> ages.iter().min_by(key=cost)
            10

            ```
        """
        return min(enumerate(map(key, iter(self))), key=itemgetter(1))[0]

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

    def collect[R: Collection[Any]](self, collector: Callable[[Iterator[T]], R]) -> R:
        """Transforms the `Iterator` into a collection.

        The most basic pattern in which `collect()` is used is to turn one collection into another.

        You take a collection, call `iter()` on it, do a bunch of transformations, and then `collect()` at the end.

        You specify the target `Collection` type by providing a **collector** function or type.

        This can be any `Callable` that takes an `Iterator[T]` and returns a `Collection[T]` of those types.

        This is equivalent to `Pipeable::into` at runtime, but with a few differences:

            - A narrower constraint (`Collection[Any]`) to specify the intent
            - Better performance (no args/kwargs unpacking).

        If you need to pass additional arguments, you can use [`Pipeable::into`][Pipeable.into] instead.

        Note:
            `Iter::collect` is overriden to provide `Seq` as the default **collector**.

        Args:
            collector (Callable[[Iterator[T]], R]): Function|type that defines the target collection. `R` is constrained to a `Collection`.

        Returns:
            R: A materialized `Collection` containing the collected elements.

        Example:
            ```python
            >>> from pyochain import Iter, Range, Vec, Dict
            >>> data = Range(0, 5)
            >>> data.iter().collect(list)
            [0, 1, 2, 3, 4]
            >>> data.iter().collect(Vec)
            Vec(0, 1, 2, 3, 4)
            >>> data.iter().map(str).enumerate().collect(Dict)
            Dict(0: '0', 1: '1', 2: '2', 3: '3', 4: '4')

            ```
            Sometimes type checkers can't infer the type of the collector, in which case you can use an explicit type annotation to help them out.

            In the example below, without the annotation in `collect()`,

            BasedPyright infer `data` as `Seq[Result[int, Any] | Result[Any, int]]` because of the conditional expression in the `map()`, which is not very useful.
            ```python
            >>> from pyochain import Range, Seq, Ok, Err, Result
            >>> data = (
            ...     Range(0, 5)
            ...     .iter()
            ...     .map(lambda x: Ok(x) if x % 2 == 0 else Err(x))
            ...     .collect(Seq[Result[int, int]])
            ... )
            >>> data
            Seq(Ok(0), Err(1), Ok(2), Err(3), Ok(4))

            ```
            Strictly speaking, this is equivalent to annotating the variable at the beginning, but some may prefer this style to keep the type information close to the actual collection operation.

            This notably avoid repetition if you collect anything else than the default `Seq` type.
        """
        return collector(iter(self))

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
            >>> a.iter().collect_into(vec).len() == vec.len()
            True
            >>> a.iter().collect_into(vec).len() == vec.len()
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

    def tail(self, n: int) -> Deque[T]:
        """Return a `Deque` of the last **n** elements of the `Iterator`.

        Args:
            n (int): Number of elements to return.

        Returns:
            Deque[T]: A `Deque` containing the last **n** elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).tail(2)
            Deque([2, 3], maxlen=2)

            ```
        """
        from collections import deque

        from ..collections import Deque

        # TODO: we should move this to Rust and make it fully lazy.
        return Deque.from_ref(deque(iter(self), n))

    def partition(self, predicate: Callable[[T], bool]) -> tuple[Vec[T], Vec[T]]:
        """Consumes the `Iterator`, creating two `Vec` from it.

        The predicate passed to `partition()` can return true, or false.

        `partition` returns a pair, all of the elements for which it returned `True`, and all of the elements for which it returned `False`.

        Args:
            predicate (Callable[[T], bool]): Function to determine partition boundaries.

        Returns:
            tuple[Vec[T], Vec[T]]: The resulting pair of collections

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3, 4, 5)).partition(lambda x: x % 2 == 0)
            (Vec(2, 4), Vec(1, 3, 5))

            ```
        """
        from .._vec import Vec

        first, second = tls.partition(iter(self), predicate)
        return Vec.from_ref(first), Vec.from_ref(second)

    def join(self: PyoIterable[str], sep: str) -> str:
        """Join all elements of the `Iterator` into a single `str`, with a specified separator.

        This is equivalent to the built-in `str.join()` method, but as a method on the `Iterator` itself.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(("a", "b", "c")).join("-")
            'a-b-c'

            ```
        """
        return sep.join(iter(self))

    def sum[U: int | bool](self: PyoIterable[U], start: int = 0) -> int:
        """Return the sum of the `Iterator`.

        If the `Iterator` is empty (i.e., yields no elements), return the value of `start` (which defaults to `0`).

        Args:
            start (int): The value to return if the `Iterator` is empty.

        Returns:
            int: The sum of all elements.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).sum()
            6
            >>> Iter(()).sum()
            0
            >>> Iter(()).sum(10)
            10

            ```
        """
        return sum(iter(self), start)

    def min[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the minimum of the `Iterator`.

        The elements of the `Iterator` must support comparison operations.

        For comparing elements using a custom **key** function, use [`min_by`][min_by] instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((3, 1, 2)).min()
            1

            ```
        """
        return min(iter(self))

    def min_by[U: SupportsRichComparison[Any]](self, key: Callable[[T], U]) -> T:
        """Return the minimum element of the `Iterator` using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            ...     is_student: bool
            ...
            ...     def get_discount(self) -> float:
            ...         return 0.1 if self.is_student else 0.0
            >>>
            >>> alice = Person("Alice", 30, False)
            >>> bob = Person("Bob", 22, True)
            >>> charlie = Person("Charlie", 25, False)
            >>> persons = Seq((alice, bob, charlie))
            >>>
            >>> persons.iter().min_by(lambda p: p.age).name
            'Bob'
            >>> persons.iter().min_by(lambda p: p.name).name
            'Alice'
            >>> persons.iter().min_by(Person.get_discount).name
            'Alice'

            ```
        """
        return min(iter(self), key=key)

    def max[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the maximum element of the `Iterator`.

        The elements of the `Iterator` must support comparison operations.

        For comparing elements using a custom **key** function, use [`max_by`][max_by] instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((3, 1, 2)).max()
            3

            ```
        """
        return max(iter(self))

    def max_by[U: SupportsRichComparison[Any]](self, key: Callable[[T], U]) -> T:
        """Return the maximum element of the `Iterator` using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            ...     is_student: bool
            ...
            ...     def get_discount(self) -> float:
            ...         return 0.1 if self.is_student else 0.0
            >>>
            >>> alice = Person("Alice", 30, False)
            >>> bob = Person("Bob", 22, True)
            >>> charlie = Person("Charlie", 25, False)
            >>> persons = Seq((alice, bob, charlie))
            >>>
            >>> persons.iter().max_by(lambda p: p.age).name
            'Alice'
            >>> persons.iter().max_by(lambda p: p.name).name
            'Charlie'
            >>> persons.iter().max_by(Person.get_discount).name
            'Bob'

            ```
        """
        return max(iter(self), key=key)

    def unpack_into[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Unpack the `Iterator` in the provided *func*, and return the result.

        This is similar to `Pipeable::into`, but instead of passing `Self`, we pass the elements inside `Self`.

        This avoids you to do `iterator.into(lambda x: (*x))`, improving performance and readability.

        Note:
            This method will consume the `Iterator`.

        Args:
            func (Callable[Concatenate[T, P], R]): Function to call with the unpacked elements of the `Iterator`.
            *args (P.args): Additional positional arguments to pass to *func*
            **kwargs (P.kwargs): Additional keyword arguments to pass to *func*

        Returns:
            R: The result of calling *func* with the unpacked elements of the `Iterator` and any additional arguments.

        Example:
            ```python
            >>> from pyochain import Seq

            >>> data = Seq((1, 2, 3))
            >>> def foo(*a: int, x: str) -> str:
            ...     return x + str(sum(a))
            >>> data.iter().unpack_into(foo, x="Result: ")
            'Result: 6'
            >>> # The example below will work, but is not type safe, as the unpacked elements are passed as explicit positional arguments.
            >>> data.iter().unpack_into(lambda a, b, c: a + b + c)
            6

            ```
        """
        return func(*iter(self), *args, **kwargs)

    def all_unique_by[U](self, key: Callable[[T], U]) -> bool:
        """Returns True if all the elements of **self** transformed by **key** are unique.

        The function returns as soon as the first non-unique element is encountered.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], U]): Function to transform items before comparison.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter("ABCb").all_unique()
            True
            >>> Iter("ABCb").all_unique_by(str.lower)
            False

            ```
        """
        return tls.all_unique_by(iter(self), key)
