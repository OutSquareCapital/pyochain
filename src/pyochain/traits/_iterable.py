from __future__ import annotations

import functools
import itertools
from collections.abc import (
    Callable,
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MappingView,
    MutableMapping,
    MutableSequence,
    Sequence,
    ValuesView,
)
from collections.abc import Set as AbstractSet
from operator import itemgetter, lt
from typing import TYPE_CHECKING, Any, Concatenate, Self, overload

import cytoolz as cz

from .._types import SupportsComparison, SupportsRichComparison
from ..rs import NONE, Checkable, Err, Ok, Option, Pipeable, Result, Some

if TYPE_CHECKING:
    from random import Random

    from .._iter import Iter

type Comparable[T] = list[T] | tuple[T, ...] | set[T] | frozenset[T]
type Comparator[T] = Callable[[Comparable[T], Comparable[T]], bool]


class PyoIterable[T](Pipeable, Checkable, Iterable[T]):
    """Base trait for all pyochain `Iterables`.

    `PyoIterable[T]` is the common API surface shared by:

    - eager `Collections`: `Seq`, `Vec`, `Set`, `SetMut`, `Dict`
    - lazy `Iterator`: `Iter`

    You typically don't instantiate this trait directly; it exists to provide a
    consistent, fluent interface across all pyochain `Iterables`.

    This is equivalent to inheriting from `collections.abc.Iterable[T]` (this
    trait already does), meaning any concrete subclass is an `Iterable[T]` as
    soon as it implements the required dunder `__iter__()`.

    On top of the standard `Iterable` protocol, it provides additional pyochain
    methods for fluent chaining and convenience (`Pipeable`, `Checkable`,
    `length()`, comparison helpers, aggregations, etc.).


    Args:
        data (Iterable[T]): The data to initialize the concrete iterable with.

    Raises:
        TypeError: Always raised when instantiating `PyoIterable` directly.
    """

    __slots__ = ()

    def __init__(self, data: Iterable[T]) -> None:  # noqa: ARG002
        msg = f"Cannot instantiate {self.__class__.__name__} directly. "
        raise TypeError(msg)

    @classmethod
    def new(cls) -> Self:
        """Create an empty `Iterable`.

        Make sure to specify the type when calling this method, e.g., `Vec[int].new()`.

        Otherwise, `T` will be inferred as `Any`.

        This can be very useful for mutable collections like `Vec` and `Dict`.

        However, this can be handy for immutable collections too, for example for representing failure steps in a pipeline.

        Returns:
            Self: A new empty `Iterable` instance.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Vec[int].new()
        >>> data
        Vec()
        >>> # Equivalent to
        >>> data: list[str] = []
        >>> data
        []
        >>> my_dict = pc.Dict[str, int].new()
        >>> my_dict.insert("a", 1)
        NONE
        >>> my_dict
        Dict('a': 1)

        ```
        """
        return cls(())

    def iter(self) -> Iter[T]:
        """Get an `Iter` over the `Iterable`.

        Call this to switch to lazy evaluation.

        Note:
            Calling this method on a class who is itself an `Iterator` has no effect.

        Returns:
            Iter[T]: An `Iterator` over the `Iterable`. The element type is inferred from the actual subclass.
        """
        from .._iter import Iter

        return Iter(self)

    def length(self) -> int:
        """Return the length of the `Iterable`.

        Returns:
            int: The count of elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).length()
        2
        >>> pc.Iter(range(5)).length()
        5

        ```
        """
        return cz.itertoolz.count(self)

    def join(self: PyoIterable[str], sep: str) -> str:
        """Join all elements of the `Iterable` into a single `str`, with a specified separator.

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
        return sep.join(self)

    def first(self) -> T:
        """Return the first element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The first element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9]).first()
        9

        ```
        """
        return cz.itertoolz.first(self)

    def second(self) -> T:
        """Return the second element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The second element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9, 8]).second()
        8

        ```
        """
        return cz.itertoolz.second(self)

    def last(self) -> T:
        """Return the last element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The last element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([7, 8, 9]).last()
        9

        ```
        """
        return cz.itertoolz.last(self)

    def sum[U: int | bool](self: PyoIterable[U]) -> int:
        """Return the sum of the `Iterable`.

        If the `Iterable` is empty, return 0.

        Returns:
            int: The sum of all elements.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).sum()
        6

        ```
        """
        return sum(self)

    def min[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the minimum of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `min_by()` instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).min()
        1

        ```
        """
        return min(self)

    def min_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the minimum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> pc.Seq([Foo(2, "a"), Foo(1, "b"), Foo(4, "c")]).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')
        >>> pc.Seq([Foo(2, "a"), Foo(1, "b"), Foo(1, "c")]).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')

        ```
        """
        return min(self, key=key)

    def max[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the maximum element of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `max_by()` instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).max()
        3

        ```
        """
        return max(self)

    def max_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the maximum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> pc.Seq([Foo(2, "a"), Foo(3, "b"), Foo(4, "c")]).max_by(key=lambda f: f.x)
        Foo(x=4, y='c')
        >>> pc.Seq([Foo(2, "a"), Foo(3, "b"), Foo(3, "c")]).max_by(key=lambda f: f.x)
        Foo(x=3, y='b')

        ```
        """
        return max(self, key=key)

    def all(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if every element of the `Iterable` is truthy.

        `Iter.all()` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterable`, and if they all return true, then so does `Iter.all()`.

        If any of them return false, it returns false.

        An empty `Iterable` returns true.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

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
        if predicate is None:
            return all(self)
        return all(predicate(x) for x in self)

    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if any element of the `Iterable` is truthy.

        `Iter.any()` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterable`, and if any of them return true, then so does `Iter.any()`.
        If they all return false, it returns false.

        An empty iterator returns false.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

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
        if predicate is None:
            return any(self)
        return any(predicate(x) for x in self)


class PyoCollection[T](PyoIterable[T], Collection[T]):
    """Base trait for eager pyochain collections.

    `PyoCollection[T]` is the shared trait for concrete, eager collections:
    `Seq`, `Vec`, `Set`, `SetMut`, `Dict`.

    It extends `PyoIterable[T]` and `collections.abc.Collection[T]` and provides
    a few convenience methods like `contains()` and `repeat()`.

    This is equivalent to subclassing `collections.abc.Collection[T]` (this
    trait already does), meaning any concrete subclass must implement the
    required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`

    On top of the standard `Collection` protocol, it provides the additional
    pyochain API (from `PyoIterable`, `Pipeable`, `Checkable`, plus the helpers
    defined here).

    """

    __slots__ = ()

    def length(self) -> int:
        return len(self)

    def contains(self, value: T) -> bool:
        """Check if the `Collection` contains the specified **value**.

        This is equivalent to using the `in` keyword directly on the `Collection`.

        Args:
            value (T): The value to check for existence.

        Returns:
            bool: True if the value exists in the Collection, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.contains(1)
        True
        >>> data.contains(3)
        False

        ```
        """
        return value in self

    def repeat(self, n: int | None = None) -> Iter[Self]:
        """Repeat the entire `Collection` **n** times (as elements) in an `Iter`.

        If **n** is `None`, repeat indefinitely.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            `Iter.cycle()` to repeat the *elements* of the `Iter` indefinitely.

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[Self]: An `Iter` of repeated `Iter`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).repeat(3).collect()
        Seq(Seq(1, 2), Seq(1, 2), Seq(1, 2))
        >>> pc.Seq(("a", "b")).repeat(2).collect()
        Seq(Seq('a', 'b'), Seq('a', 'b'))
        >>> pc.Seq([0]).repeat().flatten().take(5).collect()
        Seq(0, 0, 0, 0, 0)

        ```
        """
        from .._iter import Iter

        if n is None:
            return Iter(itertools.repeat(self))
        return Iter(itertools.repeat(self, n))

    def is_empty(self) -> bool:
        """Returns `True` if the `Collection` contains no elements.

        Returns:
            bool: `True` if the `Collection` is empty, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = pc.Dict.new()
        >>> d.is_empty()
        True
        >>> d.insert(1, "a")
        NONE
        >>> d.is_empty()
        False

        ```
        """
        return len(self) == 0


class PyoIterator[T](PyoIterable[T], Iterator[T]):
    """Base trait for lazy `Iterator` classes.

    Pyochain's `Iter[T]` implements this trait.
    This trait extends `PyoIterable[T]` and `collections.abc.Iterator[T]`, providing
    additional methods for working with lazy sequences.
    """

    __slots__ = ()

    def nth(self, n: int) -> Option[T]:
        """Return the nth item of the `Iterable` at the specified *n*.

        This is similar to `__getitem__` but for lazy `Iterators`.

        If *n* is out of bounds, returns `NONE`.

        Args:
            n (int): The index of the item to retrieve.

        Returns:
            Option[T]: `Some(item)` at the specified *n*.

        ```python
        >>> import pyochain as pc
        >>> pc.Iter([10, 20]).nth(1)
        Some(20)
        >>> pc.Iter([10, 20]).nth(3)
        NONE

        ```
        """
        try:
            return Some(next(itertools.islice(self, n, n + 1)))
        except StopIteration:
            return NONE

    def eq(self, other: Iterable[T]) -> bool:
        """Check if two `Iterable`s are equal based on their data.

        Note:
            This will consume any `Iterator` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Iterable[T]): Another instance of `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data are equal, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).eq(pc.Seq([1,2]))
        False
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2)))
        False
        >>> pc.Iter((1,2,3)).eq(pc.Vec([1,2,3]))
        True

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel or b is sentinel or a != b:
                return False
        return True

    def ne(self, other: Iterable[T]) -> bool:
        """Check if this `Iterator` and *other* are not equal based on their data.

        Args:
            other (Iterable[T]): Another instance of `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data are not equal, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2)))
        True
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2,3)))
        False

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel or b is sentinel or a != b:
                return True
        return False

    def le(self, other: Iterable[T]) -> bool:
        """Check if this `Iterator` is less than or equal to *other* based on their data.

        Args:
            other (Iterable[T]): Another instance of `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data of self is less than or equal to that of other, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2)).le(pc.Seq((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).le(pc.Seq((1,2)))
        False

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel:
                return True
            if b is sentinel:
                return False
            if a != b:
                return a < b  # type: ignore[operator]
        return True

    def lt(self, other: Iterable[T]) -> bool:
        """Check if this `Iterator` is less than *other* based on their data.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data of self is less than that of other, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2)).lt(pc.Seq((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).lt(pc.Seq((1,2)))
        False

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel:
                return True
            if b is sentinel:
                return False
            if a != b:
                return a < b  # type: ignore[operator]
        return False

    def gt(self, other: Iterable[T]) -> bool:
        """Check if this `Iterator` is greater than *other* based on their data.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data of **self** is greater than that of **other**, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).gt((1,2))
        True
        >>> pc.Iter((1,2)).gt((1,2,3))
        False

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel:
                return False
            if b is sentinel:
                return True
            if a != b:
                return a > b  # type: ignore[operator]
        return False

    def ge(self, other: Iterable[T]) -> bool:
        """Check if this `Iterator` is greater than or equal to *other* based on their data.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if the underlying data of **self** is greater than or equal to that of **other**, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).ge((1,2))
        True
        >>> pc.Iter((1,2)).ge((1,2,3))
        False

        ```
        """
        sentinel = object()
        for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
            if a is sentinel:
                return False
            if b is sentinel:
                return True
            if a != b:
                return a > b  # type: ignore[operator]
        return True

    def next(self) -> Option[T]:
        """Return the next element in the `Iterator`.

        Note:
            The actual `.__next__()` method must be conform to the Python `Iterator` Protocol, and is what will be actually called if you iterate over the `PyoIterator` instance.

            `PyoIterator.next()` is a convenience method that wraps the result in an `Option` to handle exhaustion gracefully, for custom use cases.

        Returns:
            Option[T]: The next element in the iterator. `Some[T]`, or `NONE` if the iterator is exhausted.

        Example:
        ```python
        >>> import pyochain as pc
        >>> it = pc.Seq([1, 2, 3]).iter()
        >>> it.next().unwrap()
        1
        >>> it.next().unwrap()
        2

        ```
        """
        return Option(next(self, None))

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Apply a function of two arguments cumulatively to the items of an iterable, from left to right.

        Args:
            func (Callable[[T, T], T]): Function to apply cumulatively to the items of the iterable.

        Returns:
            T: Single value resulting from cumulative reduction.

        This effectively reduces the `Iterator` to a single value.

        If initial is present, it is placed before the items of the `Iterator` in the calculation.

        It then serves as a default when the `Iterator` is empty.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).reduce(lambda a, b: a + b)
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
            This is similar to `reduce()` but with an initial value, making it equivalent to
            Python `functools.reduce()` with an initializer.

        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).fold(0, lambda acc, x: acc + x)
        6
        >>> pc.Iter([1, 2, 3]).fold(10, lambda acc, x: acc + x)
        16
        >>> pc.Iter(['a', 'b', 'c']).fold('', lambda acc, x: acc + x)
        'abc'

        ```
        """
        return functools.reduce(func, self, init)

    def find(self, predicate: Callable[[T], bool]) -> Option[T]:
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
        >>> pc.Iter(range(10)).find(predicate=gt_five)
        Some(6)
        >>> pc.Iter(range(10)).find(predicate=gt_nine).unwrap_or("missing")
        'missing'

        ```
        """
        return Option(next(filter(predicate, self), None))

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
        >>> import pyochain as pc
        >>> def is_even(x: int) -> pc.Result[bool, str]:
        ...     return pc.Ok(x % 2 == 0) if x >= 0 else pc.Err("negative number")
        >>>
        >>> pc.Iter(range(1, 6)).try_find(is_even)
        Ok(Some(2))

        ```
        """
        for item in self:
            result = predicate(item)
            if result.is_ok():
                if result.unwrap():
                    return Ok(Some(item))
            else:
                return Err(result.unwrap_err())
        return Ok(NONE)

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
        >>> import pyochain as pc
        >>> def checked_add(acc: int, x: int) -> pc.Result[int, str]:
        ...     new_val = acc + x
        ...     if new_val > 100:
        ...         return pc.Err("overflow")
        ...     return pc.Ok(new_val)
        >>>
        >>> pc.Iter([1, 2, 3]).try_fold(0, checked_add)
        Ok(6)
        >>> pc.Iter([50, 40, 20]).try_fold(0, checked_add)
        Err('overflow')
        >>> pc.Iter([]).try_fold(0, checked_add)
        Ok(0)

        ```
        """
        accumulator = init

        for item in self:
            result = func(accumulator, item)
            if result.is_ok():
                accumulator = result.unwrap()
            else:
                return result

        return Ok(accumulator)

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
        >>> import pyochain as pc
        >>> def checked_add(x: int, y: int) -> pc.Result[int, str]:
        ...     if x + y > 100:
        ...         return pc.Err("overflow")
        ...     return pc.Ok(x + y)
        >>>
        >>> pc.Iter([1, 2, 3]).try_reduce(checked_add)
        Ok(Some(6))
        >>> pc.Iter([50, 60]).try_reduce(checked_add)
        Err('overflow')
        >>> pc.Iter([]).try_reduce(checked_add)
        Ok(NONE)

        ```
        """
        first = next(self, None)

        if first is None:
            return Ok(NONE)

        accumulator: T = first

        for item in self:
            result = func(accumulator, item)
            if result.is_ok():
                accumulator = result.unwrap()
            else:
                return Err(result.unwrap_err())

        return Ok(Some(accumulator))

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3, 4, 5]).is_sorted()
        True

        ```
        If strict, tests for strict sorting, that is, returns False if equal elements are found:
        ```python
        >>> pc.Iter([1, 2, 2]).is_sorted()
        True
        >>> pc.Iter([1, 2, 2]).is_sorted(strict=True)
        False

        ```

        """
        a, b = itertools.tee(self)
        next(b, None)
        if reverse:
            b, a = a, b
        match strict:
            case True:
                return all(map(lt, a, b))
            case False:
                return not any(map(lt, b, a))

    def is_sorted_by(
        self,
        key: Callable[[T], SupportsComparison[Any]],
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
        >>> import pyochain as pc
        >>> pc.Iter(["1", "2", "3", "4", "5"]).is_sorted_by(int)
        True
        >>> pc.Iter(["5", "4", "3", "1", "2"]).is_sorted_by(int, reverse=True)
        False

        If strict, tests for strict sorting, that is, returns False if equal elements are found:
        ```python
        >>> pc.Iter(["1", "2", "2"]).is_sorted_by(int)
        True
        >>> pc.Iter(["1", "2", "2"]).is_sorted_by(key=int, strict=True)
        False

        ```
        """
        a, b = itertools.tee(map(key, self))
        next(b, None)
        if reverse:
            b, a = a, b
        match strict:
            case True:
                return all(map(lt, a, b))
            case False:
                return not any(map(lt, b, a))

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
        >>> import pyochain as pc
        >>> pc.Iter("AaaA").all_equal(key=str.casefold)
        True
        >>> pc.Iter([1, 2, 3]).all_equal(key=lambda x: x < 10)
        True

        ```
        """
        iterator = itertools.groupby(self, key)
        for _first in iterator:
            for _second in iterator:
                return False
            return True
        return True

    def all_unique[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Returns True if all the elements of iterable are unique.

        The function returns as soon as the first non-unique element is encountered.

        `Iters` with a mix of hashable and unhashable items can be used, but the function will be slower for unhashable items.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCB").all_unique()
        False
        >>> pc.Iter("ABCb").all_unique()
        True
        >>> pc.Iter("ABCb").all_unique(str.lower)
        False

        ```
        """
        seenset: set[T | U] = set()
        seenset_add = seenset.add
        seenlist: list[T | U] = []
        seenlist_add = seenlist.append
        for element in map(key, self) if key else self:
            try:
                if element in seenset:
                    return False
                seenset_add(element)
            except TypeError:
                if element in seenlist:
                    return False
                seenlist_add(element)
        return True

    def argmax[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator`.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Iter("abcdefghabcd").argmax()
        7
        >>> pc.Iter([0, 1, 2, 3, 3, 2, 1, 0]).argmax()
        3

        ```
        For example, identify the best machine learning model:
        ```python
        >>> models = pc.Seq(["svm", "random forest", "knn", "naïve bayes"])
        >>> accuracy = pc.Seq([68, 61, 84, 72])
        >>> # Most accurate model
        >>> models.get(accuracy.iter().argmax()).unwrap()
        'knn'
        >>>
        >>> # Best accuracy
        >>> accuracy.max()
        84

        ```
        """
        it = self
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

        ```python
        >>> import pyochain as pc
        >>> # Example 1: Basic usage
        >>> pc.Iter("efghabcdijkl").argmin()
        4
        >>> pc.Iter([3, 2, 1, 0, 4, 2, 1, 0]).argmin()
        3
        >>> # Example 2: look up a label corresponding to the position of a value that minimizes a cost function
        >>> def cost(x: int) -> float:
        ...     "Days for a wound to heal given a subject's age."
        ...     return x**2 - 20 * x + 150
        >>>
        >>> labels = pc.Seq(["homer", "marge", "bart", "lisa", "maggie"])
        >>> ages = pc.Seq([35, 30, 10, 9, 1])
        >>> # Fastest healing family member
        >>> labels.get(ages.iter().argmin(key=cost)).unwrap()
        'bart'
        >>> # Age with fastest healing
        >>> ages.min_by(key=cost)
        10

        ```
        """
        it = self
        if key is not None:
            it = map(key, it)
        return min(enumerate(it), key=itemgetter(1))[0]

    def take_while(self, predicate: Callable[[T], bool]) -> Self:
        """Take items while predicate holds.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Self: An `Iterator` of the items taken while the predicate is true.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 0)).take_while(lambda x: x > 0).collect()
        Seq(1, 2)

        ```
        """
        return self.__class__(itertools.takewhile(predicate, self))

    def skip_while(self, predicate: Callable[[T], bool]) -> Self:
        """Drop items while predicate holds.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Self: An `Iterator` of the items after skipping those for which the predicate is true.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 0)).skip_while(lambda x: x > 0).collect()
        Seq(0,)

        ```
        """
        return self.__class__(itertools.dropwhile(predicate, self))

    def compress(self, *selectors: bool) -> Self:
        """Filter elements using a boolean selector iterable.

        Args:
            *selectors (bool): Boolean values indicating which elements to keep.

        Returns:
            Self: An `Iterator` of the items selected by the boolean selectors.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCDEF").compress(1, 0, 1, 0, 1, 1).collect()
        Seq('A', 'C', 'E', 'F')

        ```
        """
        return self.__class__(itertools.compress(self, selectors))

    def unique(self, key: Callable[[T], Any] | None = None) -> Self:
        """Return only unique elements of the iterable.

        Args:
            key (Callable[[T], Any] | None): Function to transform items before comparison.

        Returns:
            Self: An `Iterator` of the unique items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).unique().collect()
        Seq(1, 2, 3)
        >>> pc.Iter([1, 2, 1, 3]).unique().collect()
        Seq(1, 2, 3)

        ```
        Uniqueness can be defined by key keyword
        ```python
        >>> pc.Iter(["cat", "mouse", "dog", "hen"]).unique(key=len).collect()
        Seq('cat', 'mouse')

        ```
        """
        return self.__class__(cz.itertoolz.unique(self, key=key))

    def take(self, n: int) -> Self:
        """Creates an iterator that yields the first n elements, or fewer if the underlying iterator ends sooner.

        `Iter.take(n)` yields elements until n elements are yielded or the end of the iterator is reached (whichever happens first).

        The returned iterator is either:

        - A prefix of length n if the original iterator contains at least n elements
        - All of the (fewer than n) elements of the original iterator if it contains fewer than n elements.

        Args:
            n (int): Number of elements to take.

        Returns:
            Self: An `Iterator` of the first n items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = [1, 2, 3]
        >>> pc.Iter(data).take(2).collect()
        Seq(1, 2)
        >>> pc.Iter(data).take(5).collect()
        Seq(1, 2, 3)

        ```
        """
        return self.__class__(itertools.islice(self, n))

    def skip(self, n: int) -> Self:
        """Drop first n elements.

        Args:
            n (int): Number of elements to skip.

        Returns:
            Self: An `Iterator` of the items after skipping the first n items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 3)).skip(1).collect()
        Seq(2, 3)

        ```
        """
        return self.__class__(cz.itertoolz.drop(n, self))

    def step_by(self, step: int) -> Self:
        """Creates an `Iterator` starting at the same point, but stepping by the given **step** at each iteration.

        Note:
            The first element of the iterator will always be returned, regardless of the **step** given.

        Args:
            step (int): Step size for selecting items.

        Returns:
            Self: An `Iterator` of every nth item.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([0, 1, 2, 3, 4, 5]).step_by(2).collect()
        Seq(0, 2, 4)

        ```
        """
        return self.__class__(cz.itertoolz.take_nth(step, self))

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self:
        """Return a slice of the `Iterator`.

        Args:
            start (int | None): Starting index of the slice.
            stop (int | None): Ending index of the slice.
            step (int | None): Step size for the slice.

        Returns:
            Self: An `Iterator` of the sliced items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = (1, 2, 3, 4, 5)
        >>> pc.Iter(data).slice(1, 4).collect()
        Seq(2, 3, 4)
        >>> pc.Iter(data).slice(step=2).collect()
        Seq(1, 3, 5)

        ```
        """
        return self.__class__(itertools.islice(self, start, stop, step))

    def cycle(self) -> Self:
        """Repeat the `Iterator` indefinitely.

        Warning:
            This creates an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            `Iter.repeat()` to repeat *self* as elements (`Iter[Self]`).

        Returns:
            Self: A new `Iterator` that cycles through the elements indefinitely.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).cycle().take(5).collect()
        Seq(1, 2, 1, 2, 1)

        ```
        """
        return self.__class__(itertools.cycle(self))

    def intersperse(self, element: T) -> Self:
        """Creates a new `Iterator` which places a copy of separator between adjacent items of the original iterator.

        Args:
            element (T): The element to interpose between items.

        Returns:
            Self: A new `Iterator` with the element interposed.

        Example:
        ```python
        >>> import pyochain as pc
        >>> # Simple example with numbers
        >>> pc.Iter([1, 2, 3]).intersperse(0).collect()
        Seq(1, 0, 2, 0, 3)
        >>> # Useful when chaining with other operations
        >>> pc.Iter([10, 20, 30]).intersperse(5).sum()
        70
        >>> # Inserting separators between groups, then flattening
        >>> pc.Iter([[1, 2], [3, 4], [5, 6]]).intersperse([-1]).flatten().collect()
        Seq(1, 2, -1, 3, 4, -1, 5, 6)

        ```
        """
        return self.__class__(cz.itertoolz.interpose(element, self))

    def random_sample(
        self, probability: float, state: Random | int | None = None
    ) -> Self:
        """Return elements from the `Iterator` with a given *probability*.

        `.random_sample()` considers each item independently and without replacement.

        See below how the first time it returned 13 items and the next time it returned 6 items.

        Args:
            probability (float): The probability of including each element.
            state (Random | int | None): Random state or seed for deterministic sampling.

        Returns:
            Self: A new `Iterator` with randomly sampled elements.
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
        return self.__class__(
            cz.itertoolz.random_sample(probability, self, random_state=state)
        )

    def insert(self, value: T) -> Self:
        """Prepend the *value* to the `Iterator`.

        Note:
            This can be considered the equivalent as `list.append()`, but for a lazy `Iterator`.
            However, append add the value at the **end**, while insert add it at the **beginning**.

        See Also:
            `Iter.chain()` to add multiple elements at the end of the `Iterator`.

        Args:
            value (T): The value to prepend.

        Returns:
            Self: A new Iterable wrapper with the value prepended.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((2, 3)).insert(1).collect()
        Seq(1, 2, 3)

        ```
        """
        return self.__class__(cz.itertoolz.cons(value, self))

    def interleave(self, *others: Iterable[T]) -> Self:
        """Interleave multiple sequences element-wise.

        Args:
            *others (Iterable[T]): Other iterables to interleave.

        Returns:
            Self: A new Iterable wrapper with interleaved elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).interleave((3, 4)).collect()
        Seq(1, 3, 2, 4)

        ```
        """
        return self.__class__(cz.itertoolz.interleave((self, *others)))

    def chain(self, *others: Iterable[T]) -> Self:
        """Concatenate **self** with one or more `Iterables`, any of which may be infinite.

        In other words, it links **self** and **others** together, in a chain. 🔗

        An infinite `Iterable` will prevent the rest of the arguments from being included.

        This is equivalent to `list.extend()`, except it is fully lazy and works with any `Iterable`.

        See Also:
            `Iter.insert()` to add a single element at the beginning of the `Iterator`.

        Args:
            *others (Iterable[T]): Other iterables to concatenate.

        Returns:
            Self: A new `Iterator` which will first iterate over values from the original `Iterator` and then over values from the **others** `Iterable`s.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).chain((3, 4), [5]).collect()
        Seq(1, 2, 3, 4, 5)
        >>> pc.Iter((1, 2)).chain(pc.Iter.from_count(3)).take(5).collect()
        Seq(1, 2, 3, 4, 5)

        ```
        """
        return self.__class__(cz.itertoolz.concat((self, *others)))

    def elements(self) -> Self:
        """Iterator over elements repeating each as many times as its count.

        Note:
            if an element's count has been set to zero or is a negative
            number, elements() will ignore it.

        Returns:
            Self: A new `Iterator` with elements repeated according to their counts.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCABC").elements().sort()
        Vec('A', 'A', 'B', 'B', 'C', 'C')

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

        return self.__class__(Counter(self).elements())

    def accumulate(self, func: Callable[[T, T], T], initial: T | None = None) -> Self:
        """Return an `Iterator` of accumulated binary function results.

        In principle, `.accumulate()` is similar to `.fold()` if you provide it with the same binary function.

        However, instead of returning the final accumulated result, it returns an `Iterator` that yields the current value `T` of the accumulator for each iteration.

        In other words, the last element yielded by `.accumulate()` is what would have been returned by `.fold()` if it had been used instead.

        Args:
            func (Callable[[T, T], T]): A binary function to apply cumulatively.
            initial (T | None): Optional initial value to start the accumulation.

        Returns:
            Self: A new `Iterator` with accumulated results.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 3)).accumulate(lambda a, b: a + b, 0).collect()
        Seq(0, 1, 3, 6)
        >>> # The final accumulated result is the same as fold:
        >>> pc.Iter((1,2,3)).fold(0, lambda a, b: a + b)
        6
        >>> pc.Iter((1, 2, 3)).accumulate(lambda a, b: a * b).collect()
        Seq(1, 2, 6)


        ```
        """
        return self.__class__(itertools.accumulate(self, func, initial=initial))

    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
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

        Returns:
            None: This is a terminal operation with no return value.


        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).iter().for_each(lambda x: print(x + 1))
        2
        3
        4

        ```
        """
        for v in self:
            func(v, *args, **kwargs)

    @overload
    def for_each_star[R](
        self: PyoIterator[tuple[Any]],
        func: Callable[[Any], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, R](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[[T1, T2], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, R](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, R](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
    ) -> None: ...
    def for_each_star[U: Iterable[Any], R](
        self: PyoIterator[U],
        func: Callable[..., R],
    ) -> None:
        """Consume the `Iterator` by applying a function to each unpacked item in the `Iterable` element.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Each item yielded by the `Iterator` is expected to be an `Iterable` itself (e.g., a tuple or list),
        and its elements are unpacked as arguments to the provided function.

        This is often used after methods like `zip()` or `enumerate()` that yield tuples.

        Args:
            func (Callable[..., R]): Function to apply to each unpacked element.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([(1, 2), (3, 4)]).for_each_star(lambda x, y: print(x + y))
        3
        7

        ```
        """
        for item in self:
            func(*item)

    def try_for_each[E](self, f: Callable[[T], Result[Any, E]]) -> Result[None, E]:
        """Applies a fallible function to each item in the `Iterator`, stopping at the first error and returning that error.

        This can also be thought of as the fallible form of `.for_each()`.

        Args:
            f (Callable[[T], Result[Any, E]]): A function that takes an item of type `T` and returns a `Result`.

        Returns:
            Result[None, E]: Returns `Ok(None)` if all applications of **f** were successful (i.e., returned `Ok`), or the first error `E` encountered.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def validate_positive(n: int) -> pc.Result[None, str]:
        ...     if n > 0:
        ...         return pc.Ok(None)
        ...     return pc.Err(f"Value {n} is not positive")
        >>> pc.Iter([1, 2, 3, 4, 5]).try_for_each(validate_positive)
        Ok(None)
        >>> # Short-circuit on first error:
        >>> pc.Iter([1, 2, -1, 4]).try_for_each(validate_positive)
        Err('Value -1 is not positive')

        ```
        """
        for item in self:
            res = f(item)
            if res.is_err():
                return res
        return Ok(None)


class PyoSequence[T](PyoCollection[T], Sequence[T]):
    """Base trait for eager pyochain sequences.

    `PyoSequence[T]` is the shared trait for concrete, eager sequence-like
    collections: `Seq` and `Vec`.

    It extends `PyoCollection[T]` and `collections.abc.Sequence[T]`.

    This is equivalent to subclassing `collections.abc.Sequence[T]` (this trait
    already does), meaning any concrete subclass must implement the required
    `Sequence` dunder methods:

    - `__getitem__`
    - `__len__`
    - `__contains__`
    - `__iter__`

    On top of the standard `Sequence` protocol, it provides the additional
    pyochain API (from `PyoCollection`, `Pipeable`, `Checkable`, plus any helpers defined here).

    """

    __slots__ = ()

    @overload
    def get(self, index: int) -> Option[T]: ...
    @overload
    def get(self, index: slice) -> Option[Sequence[T]]: ...
    def get(self, index: int | slice) -> Option[T] | Option[Sequence[T]]:
        """Return the element at the specified index as `Some(value)`, or `None` if the index is out of bounds.

        Args:
            index (int | slice): The index or slice of the element to retrieve.

        Returns:
            Option[T] | Option[Sequence[T]]: `Some(value)` if the index is valid, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([10, 20, 30]).get(1)
        Some(20)
        >>> pc.Seq([10, 20, 30]).get(5)
        NONE

        ```
        """
        try:
            return Some(self.__getitem__(index))  # pyright: ignore[reportReturnType]
        except IndexError:
            return NONE

    def rev(self) -> Iter[T]:
        """Return an `Iterator` with the elements of the `Sequence` in reverse order.

        Returns:
            Iter[T]: An `Iterator` with the elements in reverse order.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).rev().collect()
        Seq(3, 2, 1)

        ```
        """
        from .._iter import Iter

        return Iter(reversed(self))

    def is_distinct(self) -> bool:
        """Return True if all items of the `Sequence` are distinct.

        Returns:
            bool: True if all items are distinct, False otherwise.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).is_distinct()
        True

        ```
        """
        return cz.itertoolz.isdistinct(self)


class PyoSet[T](PyoCollection[T], AbstractSet[T]):
    """Base trait for eager pyochain set-like collections.

    `PyoSet[T]` is the shared trait for concrete, eager set-like
    collections: `Set` and `FrozenSet`.

    It extends `PyoCollection[T]` and `collections.abc.Set[T]`.

    This is equivalent to subclassing `collections.abc.Set[T]` (this trait
    already does), meaning any concrete subclass must implement the required
    `Set` dunder methods:

    - `__contains__`
    - `__iter__`
    - `__len__`

    On top of the standard `Set` protocol, it provides the additional
    pyochain API (from `PyoCollection`, `Pipeable`, `Checkable`, plus any helpers defined here).

    """

    __slots__ = ()

    def is_subset(self, other: AbstractSet[T]) -> bool:
        """Check if the `Set` is a subset of another set.

        Args:
            other (AbstractSet[T]): The other set to compare with.

        Returns:
            bool: True if the `Set` is a subset of the other set, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2}).is_subset({1, 2, 3})
        True
        >>> pc.Set({1, 4}).is_subset({1, 2, 3})
        False

        ```
        """
        return self <= other

    def is_subset_strict(self, other: AbstractSet[T]) -> bool:
        """Check if the `Set` is a proper subset of another set.

        Args:
            other (AbstractSet[T]): The other set to compare with.

        Returns:
            bool: True if the `Set` is a proper subset of the other set, False otherwise.
        """
        return self < other

    def eq(self, other: AbstractSet[T]) -> bool:
        """Check if the `Set` is equal to another set.

        Args:
            other (AbstractSet[T]): The other set to compare with.

        Returns:
            bool: True if the `Set` is equal to the other set, False otherwise.
        """
        return self == other

    def is_superset(self, other: AbstractSet[T]) -> bool:
        """Check if the `Set` is a superset of another set.

        Args:
            other (AbstractSet[T]): The other set to compare with.

        Returns:
            bool: True if the `Set` is a superset of the other set, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 3}).is_superset({1, 2})
        True
        >>> pc.Set({1, 2}).is_superset({1, 2, 3})
        False

        ```
        """
        return self >= other

    def intersection(self, other: AbstractSet[T]) -> Self:
        """Return the intersection of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to intersect with.

        Returns:
            Self: A new `Set` containing the intersection of the two sets.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).intersection({2, 3})
        Set(2,)

        ```
        """
        return self.__class__(self & other)

    def r_intersection(self, other: AbstractSet[T]) -> Self:
        """Return the intersection of another set with the `Set`.

        Args:
            other (AbstractSet[T]): The other set to intersect with.

        Returns:
            Self: A new `Set` containing the intersection of the two sets.
        """
        return self.__class__(other & self)

    def union(self, other: AbstractSet[T]) -> Self:
        """Return the union of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to unite with.

        Returns:
            Self: A new `Set` containing the union of the two sets.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).union([2, 3]).union([4]).iter().sort()
        Vec(1, 2, 3, 4)

        ```
        """
        return self.__class__(self | other)

    def r_union(self, other: AbstractSet[T]) -> Self:
        """Return the union of another set with the `Set`.

        Args:
            other (AbstractSet[T]): The other set to unite with.

        Returns:
            Self: A new `Set` containing the union of the two sets.
        """
        return self.__class__(other | self)

    def difference(self, other: AbstractSet[T]) -> Self:
        """Return the difference of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to subtract.

        Returns:
            Self: A new `Set` containing the difference of the two sets.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).difference([2, 3])
        Set(1,)

        """
        return self.__class__(self - other)

    def r_difference(self, other: AbstractSet[T]) -> Self:
        """Return the reverse difference of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to subtract from.

        Returns:
            Self: A new `Set` containing the reverse difference of the two sets.
        """
        return self.__class__(other - self)

    def symmetric_difference(self, other: AbstractSet[T]) -> Self:
        """Return the symmetric difference of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to symmetric difference with.

        Returns:
            Self: A new `Set` containing the symmetric difference of the two sets.
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2, 2}).symmetric_difference([2, 3]).iter().sort()
        Vec(1, 3)
        >>> pc.Set({1, 2, 3}).symmetric_difference([3, 4, 5]).iter().sort()
        Vec(1, 2, 4, 5)

        ```
        """
        return self.__class__(self ^ other)

    def r_symmetric_difference(self, other: AbstractSet[T]) -> Self:
        """Return the symmetric difference of the `Set` with another set.

        Args:
            other (AbstractSet[T]): The other set to symmetric difference with.

        Returns:
            Self: A new `Set` containing the symmetric difference of the two sets.


        """
        return self.__class__(other ^ self)

    def is_disjoint(self, other: AbstractSet[T]) -> bool:
        """Check if the `Set` has no elements in common with another set.

        Args:
            other (AbstractSet[T]): The other set to compare with.

        Returns:
            bool: True if the `Set` and the other set have no elements in common, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Set({1, 2}).is_disjoint({3, 4})
        True
        >>> pc.Set({1, 2}).is_disjoint({2, 3})
        False

        ```
        """
        return self.isdisjoint(other)


class PyoMappingView[T](MappingView, PyoCollection[T]):
    """Base trait for eager pyochain mapping view collections.

    `PyoMappingView[T]` is the shared trait for concrete mapping views
    returned by mapping operations.

    This trait extends both `MappingView` from `collections.abc` and `PyoCollection[T]`.
    """

    __slots__ = ()


class PyoValuesView[V](ValuesView[V], PyoMappingView[V]):
    """A view of the values in a pyochain mapping.

    This class provides a view over the values contained in a pyochain mapping, with
    additional methods from the `PyoMappingView` and `PyoCollection` traits.

    See Also:
        `PyoMapping.values()`: Method that returns this view.
    """

    __slots__ = ()


class PyoKeysView[K](KeysView[K], PyoMappingView[K], PyoSet[K]):
    """A view of the keys in a pyochain mapping.

    This class provides a view over the keys contained in a pyochain mapping, with
    additional methods from the `PyoMappingView`, `PyoSet`, and `PyoCollection` traits.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping.keys()`: Method that returns this view.
    """

    __slots__ = ()


class PyoItemsView[K, V](
    ItemsView[K, V], PyoMappingView[tuple[K, V]], PyoSet[tuple[K, V]]
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    This class provides a view over the items contained in a pyochain mapping, with
    additional methods from the `PyoMappingView`, `PyoSet`, and `PyoCollection` traits.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping.items()`: Method that returns this view.
    """

    __slots__ = ()


class PyoMapping[K, V](PyoCollection[K], Mapping[K, V]):
    """Base trait for eager pyochain immutable mapping collections.

    `PyoMapping[K, V]` is the shared trait for concrete, eager mapping-like
    collections such as `Dict`.

    It extends `PyoCollection[K]` and `collections.abc.Mapping[K, V]`.

    This is equivalent to subclassing `collections.abc.Mapping[K, V]` (this trait
    already does), meaning any concrete subclass must implement the required
    `Mapping` dunder methods:

    - `__getitem__`
    - `__iter__`
    - `__len__`

    On top of the standard `Mapping` protocol, it provides the additional
    pyochain API (from `PyoCollection`, `Pipeable`, `Checkable`, plus any helpers defined here).
    """

    __slots__ = ()

    def keys(self) -> PyoKeysView[K]:
        """Return a view of the `Mapping` keys.

        Returns:
            PyoKeysView[K]: A view of the dictionary's keys.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.keys()
        PyoKeysView(Dict(1: 'a', 2: 'b'))

        ```
        """
        return PyoKeysView(self)

    def values(self) -> PyoValuesView[V]:
        """Return a view of the `Mapping` values.

        Returns:
            PyoValuesView[V]: A view of the dictionary's values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.values()
        PyoValuesView(Dict(1: 'a', 2: 'b'))

        ```
        """
        return PyoValuesView(self)

    def items(self) -> PyoItemsView[K, V]:
        """Return a view of the `Mapping` items.

        Returns:
            PyoItemsView[K, V]: A view of the dictionary's (key, value) pairs.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.items()
        PyoItemsView(Dict(1: 'a', 2: 'b'))

        ```
        """
        return PyoItemsView(self)


class PyoMutableMapping[K, V](PyoMapping[K, V], MutableMapping[K, V]):
    """Base trait for eager pyochain mutable mapping collections.

    `PyoMutableMapping[K, V]` is the shared trait for concrete, eager mutable mapping-like
    collections such as `Dict`.

    It extends `PyoMapping[K, V]` and `collections.abc.MutableMapping[K, V]`.

    This is equivalent to subclassing `collections.abc.MutableMapping[K, V]` (this trait
    already does), meaning any concrete subclass must implement the required
    `MutableMapping` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__iter__`
    - `__len__`

    On top of the standard `MutableMapping` protocol, it provides the additional
    pyochain API (from `PyoMapping`, `PyoCollection`, `Pipeable`, `Checkable`, plus any helpers defined here).
    """

    __slots__ = ()

    def insert(self, key: K, value: V) -> Option[V]:
        """Insert a key-value pair into the `MutableMapping`.

        If the `MutableMapping` did not have this **key** present, `NONE` is returned.

        If the `MutableMapping` did have this **key** present, the **value** is updated, and the old value is returned.

        The **key** is not updated.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Option[V]: The previous value associated with the key, or None if the key was not present.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict.new()
        >>> data.insert(37, "a")
        NONE
        >>> data.is_empty()
        False

        >>> data.insert(37, "b")
        Some('a')
        >>> data.insert(37, "c")
        Some('b')
        >>> data[37]
        'c'

        ```
        """
        previous = self.get(key, None)
        self[key] = value
        return Option(previous)

    def try_insert(self, key: K, value: V) -> Result[V, KeyError]:
        """Tries to insert a key-value pair into the `MutableMapping`, and returns a `Result[V, KeyError]` containing the value in the entry (if successful).

        If the `MutableMapping` already had this **key** present, nothing is updated, and an error containing the occupied entry and the value is returned.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Result[V, KeyError]: `Ok` containing the value if the **key** was not present, or `Err` containing a `KeyError` if the **key** already existed.

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = pc.Dict.new()
        >>> d.try_insert(37, "a").unwrap()
        'a'
        >>> d.try_insert(37, "b")
        Err(KeyError('Key 37 already exists with value a.'))

        ```
        """
        if key in self:
            return Err(KeyError(f"Key {key} already exists with value {self[key]}."))
        self[key] = value
        return Ok(value)

    def remove(self, key: K) -> Option[V]:
        """Remove a **key** from the `MutableMapping` and return its value if it existed.

        Equivalent to `dict.pop(key, None)`, with an `Option` return type.

        Args:
            key (K): The key to remove.

        Returns:
            Option[V]: The value associated with the removed **key**, or `None` if the **key** was not present.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.remove(1)
        Some('a')
        >>> data.remove(3)
        NONE

        ```
        """
        return Option(self.pop(key, None))

    def remove_entry(self, key: K) -> Option[tuple[K, V]]:
        """Remove a key from the `MutableMapping` and return the item if it existed.

        Return an `Option[tuple[K, V]]` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[tuple[K, V]]: `Some((key, value))` pair associated with the removed key, or `None` if the **key** was not present.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.remove_entry(1)
        Some((1, 'a'))
        >>> data.remove_entry(3)
        NONE

        ```
        """
        return Some((key, self.pop(key))) if key in self else NONE

    def get_item(self, key: K) -> Option[V]:
        """Retrieve a value from the `MutableMapping`.

        Returns `Some(value)` if the **key** exists, or `None` if it does not.

        Args:
            key (K): The key to look up.

        Returns:
            Option[V]: `Some(value)` that is associated with the **key**, or `None` if not found.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = {"a": 1}
        >>> pc.Dict(data).get_item("a")
        Some(1)
        >>> pc.Dict(data).get_item("x").unwrap_or('Not Found')
        'Not Found'

        ```
        """
        return Option(self.get(key, None))


class PyoMutableSequence[T](PyoSequence[T], MutableSequence[T]):
    """Base trait for eager pyochain mutable sequence collections.

    `PyoMutableSequence[T]` is the shared trait for concrete, eager mutable sequence-like
    collections: `Vec`.

    It extends `PyoSequence[T]` and `collections.abc.MutableSequence[T]`.

    This is equivalent to subclassing `collections.abc.MutableSequence[T]` (this trait
    already does), meaning any concrete subclass must implement the required
    `MutableSequence` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__len__`
    - `insert`

    On top of the standard `MutableSequence` protocol, it provides the additional
    pyochain API (from `PyoSequence`, `PyoCollection`, `Pipeable`, `Checkable`), and various methods inspired from Rust's `Vec` type.

    Those methods provides memory-efficient in-place operations.

    They are slower than simple `.extend()`, slices and `clear()` calls, but avoids all intermediate allocations, making them suitable for large collections where memory usage is a concern.
    """

    __slots__ = ()

    def retain(self, predicate: Callable[[T], bool]) -> None:
        """Retains only the elements specified by the *predicate*.

        In other words, remove all elements e for which the *predicate* function returns `False`.

        This method operates in place, visiting each element exactly once in forward order, and preserves the order of the retained elements.

        Note:
            This is similar to filtering, but operates in place without allocating a new collection once collected.

            For example `new_list = list(filter(predicate, my_list))` followed by `my_list.clear()` would allocate a new collection before clearing the original, resulting in higher peak memory usage.

        Args:
            predicate (Callable[[T], bool]): A function that returns `True` for elements to keep and `False` for elements to remove.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> vec = pc.Vec([1, 2, 3, 4])
        >>> vec.retain(lambda x: x % 2 == 0)
        >>> vec
        Vec(2, 4)

        ```
        External state may be used to decide which elements to keep.

        ```python
        >>> vec = pc.Vec([1, 2, 3, 4, 5])
        >>> keep = pc.Seq([False, True, True, False, True]).iter()
        >>> vec.retain(lambda _: next(keep))
        >>> vec
        Vec(2, 3, 5)

        ```

        """
        write_idx = 0
        length = len(self)
        for read_idx in range(length):
            if predicate(self[read_idx]):
                self[write_idx] = self[read_idx]
                write_idx += 1
        pop = self.pop
        while len(self) > write_idx:
            pop(write_idx)

    def truncate(self, length: int) -> None:
        """Shortens the `MutableSequence`, keeping the first *length* elements and dropping the rest.

        If *length* is greater or equal to the `MutableSequence` current `__len__()`, this has no effect.

        The `.drain()` method can emulate `.truncate()`, but causes the excess elements to be returned instead of dropped.

        Note:
            This is equivalent to `del seq[length:]`, except that it won't create an intermediate slice object.

        Args:
            length (int): The length to truncate the `MutableSequence` to.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> # Truncating a five element vector to two elements:
        >>> vec = pc.Vec([1, 2, 3, 4, 5])
        >>> vec.truncate(2)
        >>> vec
        Vec(1, 2)

        ```
        No truncation occurs when len is greater than the `MutableSequence` current length:
        ```python
        >>> import pyochain as pc
        >>> vec = pc.Vec([1, 2, 3])
        >>> vec.truncate(8)
        >>> vec
        Vec(1, 2, 3)

        ```
        Truncating when len == 0 is equivalent to calling the clear method.
        ```python
        >>> import pyochain as pc
        >>> vec = pc.Vec([1, 2, 3])
        >>> vec.truncate(0)
        >>> vec
        Vec()

        ```
        """
        pop = self.pop
        for _ in range(len(self) - length):
            pop()

    def extend_move(self, other: Self | list[T]) -> None:
        """Moves all the elements of *other* into *self*, leaving *other* empty.

        Note:
            This is equivalent to `extend(other)` followed by `other.clear()`, but avoids intermediate allocations by moving elements one at a time.

        Args:
            other (Self | list[T]): The other `MutableSequence` to move elements from.

        Examples:
        >>> import pyochain as pc
        >>> v1 = pc.Vec([1, 2, 3])
        >>> v2 = pc.Vec([4, 5, 6])
        >>> v1.extend_move(v2)
        >>> v1
        Vec(1, 2, 3, 4, 5, 6)
        >>> v2
        Vec()
        """
        pop = functools.partial(other.pop, 0)
        self.extend(pop() for _ in range(len(other)))
