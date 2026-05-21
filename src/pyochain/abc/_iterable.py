from __future__ import annotations

import functools
import itertools
from abc import ABC
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    MappingView,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Concatenate, Self, overload, override

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from .._types import SupportsComparison, SupportsRichComparison
from ..rs import NONE, Checkable, Err, Ok, Option, Pipeable, Result, Some, option

if TYPE_CHECKING:
    from .._iter import Iter, Vec
    from .._seq import Seq
    from .._set import PyoItemsView, PyoKeysView, PyoValuesView
    from ..rs import Option, Result

type Comparable[T] = list[T] | tuple[T, ...] | set[T] | frozenset[T]
type Comparator[T] = Callable[[Comparable[T], Comparable[T]], bool]


@dataclass(slots=True)
class DrainIterator[T](Iterator[T]):
    """An `Iterator` that drains elements from a `Vec` within a specified range.

    This class is not supposed to be used directly. Use `Vec.drain()` instead to obtain an `Iter` wrapper around it.

    See `Vec.drain()` for details.
    """

    _vec: MutableSequence[T]
    _idx: int
    _end_idx: int

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> T:
        if self._idx >= self._end_idx:
            raise StopIteration
        val = self._vec.pop(self._idx)
        self._end_idx -= 1
        return val

    def __del__(self) -> None:
        pop = self._vec.pop
        while self._idx < self._end_idx:
            _ = pop(self._idx)
            self._end_idx -= 1


class PyoIterable[T](Pipeable, Checkable, Iterable[T], ABC):
    """Base ABC for all pyochain `Iterables`.

    It's the common API surface shared by:

    - eager `Collections`: `Seq`, `Vec`, `Set`, `SetMut`, `Dict`
    - lazy `Iterator`: `Iter`

    It extends the standard `Iterable[T]` protocol, as well as `Pipeable` and `Checkable`.

    All concrete subclasses must implement `__iter__()`.

    Since it's very straightforward to implement, it can very easily be integrated into business logic classes to provide them with a rich set of methods for free.

    Example:
    ```python
    >>> from pyochain.abc import PyoIterable
    >>> from dataclasses import dataclass
    >>> @dataclass(slots=True)
    ... class ClientRegistry(PyoIterable[str]):
    ...     clients: list[str]
    ...
    ...     def __iter__(self):
    ...         return iter(self.clients)
    >>>
    >>> registry = ClientRegistry(["Alice", "Bob", "Charlie"])
    >>> registry.all(lambda name: name.startswith("A"))
    False
    >>> registry.join(", ")
    'Alice, Bob, Charlie'
    >>> registry.iter().map(str.lower).join(", ")
    'alice, bob, charlie'
    >>> registry.ok_or("Registry is empty").map(lambda s: s.join(", "))
    Ok('Alice, Bob, Charlie')

    ```
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def iter(self) -> Iter[T]:
        """Get an `Iter` over the `Iterable`.

        Call this to switch to lazy evaluation.

        Note:
            Calling this method on a class who is itself an `Iterator` has no effect.

        Returns:
            Iter[T]: An `Iterator` over the `Iterable`. The element type is inferred from the actual subclass.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> seq = Seq((1, 2, 3))
        >>> iterator = seq.iter()
        >>> iterator.collect()
        Seq(1, 2, 3)
        >>> # iterator is now empty
        >>> iterator.collect()
        Seq()

        ```
        """
        from .._iter import Iter

        return Iter(iter(self))

    def unpack_into[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Unpack the `Iterable` in the provided *func*, and return the result.

        This is similar to `Pipeable::into`, but instead of passing `Self`, we pass the elements inside `Self`.

        This avoids you to do `iterable.into(lambda x: (*x))`, improving performance and readability.

        Note:
            This method, if called on a lazy `Iterator`, will consume it.

            As such, this can be considered as an alternative `Iter::collect` method.

        Args:
            func (Callable[Concatenate[T, P], R]): Function to call with the unpacked elements of the `Iterable`.
            *args (P.args): Additional positional arguments to pass to *func*
            **kwargs (P.kwargs): Additional keyword arguments to pass to *func*

        Returns:
            R: The result of calling *func* with the unpacked elements of the `Iterable` and any additional arguments.

        Example:
        ```python
        >>> from pyochain import Seq

        >>> data = Seq((1, 2, 3))
        >>> def foo(*a: int, x: str) -> str:
        ...     return x + str(sum(a))
        >>> data.unpack_into(foo, x="Result: ")
        'Result: 6'
        >>> # The example below will work, but is not type safe, as the unpacked elements are passed as explicit positional arguments.
        >>> data.unpack_into(lambda a, b, c: a + b + c)
        6

        ```
        """
        return func(*self, *args, **kwargs)

    def join(self: PyoIterable[str], sep: str) -> str:
        """Join all elements of the `Iterable` into a single `str`, with a specified separator.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq(("a", "b", "c")).join("-")
        'a-b-c'

        ```
        """
        return sep.join(iter(self))

    def first(self) -> T:
        """Return the first element of the `Iterable`.

        By default, this method convert the `Iterable` to an `Iterator` and returns the first element by calling `next()` on it.

        On `PyoSequence` and its subclasses (`Seq`, `Range`, etc.), this is overriden to directly use an efficient `__getitem__` access.

        If you already are using an `Iter`, prefer `Iter.next()` instead, which returns an `Option[T]` to handle exhaustion gracefully.

        Returns:
            T: The first element of the `Iterable`.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> data = Seq((1, 2))
        >>> data.first()
        1
        >>> iterator = data.iter()
        >>> iterator.first()
        1
        >>> iterator.first()
        2
        >>> # iterator is now empty, using first again would raise an error
        >>> iterator.next()
        NONE

        ```
        """
        return next(iter(self))

    def second(self) -> T:
        """Return the second element of the `Iterable`.

        Similar to `first()`, see its documentation for details.

        Returns:
            T: The second element of the `Iterable`.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((9, 8)).second()
        8

        ```
        """
        seq = iter(self)
        _ = next(seq)
        return next(seq)

    def last(self) -> T:
        """Return the last element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The last element of the `Iterable`.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((7, 8, 9)).last()
        9

        ```
        """
        return tls.last(iter(self))

    def length(self) -> int:
        """Return the length of the `Iterable`.

        By default, this method converts the `Iterable` to an `Iterator` and counts the elements by consuming it.

        This is overriden on `PyoCollection` and its subclasses to directly use an efficient `__len__` access.

        Returns:
            int: The count of elements.

        Example:
        ```python
        >>> from pyochain import Seq, Range, Iter
        >>> Seq((1, 2)).length()
        2
        >>> Range(0, 5).length()
        5
        >>> data = Iter((1, 2, 3))
        >>> data.length()
        3
        >>> # data is now empty
        >>> data.length()
        0

        ```
        """
        return tls.length(iter(self))

    def sum[U: int | bool](self: PyoIterable[U]) -> int:
        """Return the sum of the `Iterable`.

        If the `Iterable` is empty, return 0.

        Returns:
            int: The sum of all elements.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((1, 2, 3)).sum()
        6

        ```
        """
        return sum(iter(self))

    def min[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the minimum of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `min_by()` instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((3, 1, 2)).min()
        1

        ```
        """
        return min(iter(self))

    def min_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the minimum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> Seq((Foo(2, "a"), Foo(1, "b"), Foo(4, "c"))).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')
        >>> Seq((Foo(2, "a"), Foo(1, "b"), Foo(1, "c"))).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')

        ```
        """
        return min(iter(self), key=key)

    def max[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the maximum element of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `max_by()` instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((3, 1, 2)).max()
        3

        ```
        """
        return max(iter(self))

    def max_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the maximum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> Seq((Foo(2, "a"), Foo(3, "b"), Foo(4, "c"))).max_by(key=lambda f: f.x)
        Foo(x=4, y='c')
        >>> Seq((Foo(2, "a"), Foo(3, "b"), Foo(3, "c"))).max_by(key=lambda f: f.x)
        Foo(x=3, y='b')

        ```
        """
        return max(iter(self), key=key)

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
        >>> from pyochain import Seq
        >>> Seq((1, True)).all()
        True
        >>> Seq(()).all()
        True
        >>> Seq((1, 0)).all()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> Seq((2, 4, 6)).all(is_even)
        True

        ```
        """
        if predicate is None:
            return all(iter(self))
        return all(predicate(x) for x in iter(self))

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
        >>> from pyochain import Seq, Range
        >>> Seq((0, 1)).any()
        True
        >>> Range(0, 0).any()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> Seq((1, 3, 4)).any(is_even)
        True

        ```
        """
        if predicate is None:
            return any(iter(self))
        return any(predicate(x) for x in iter(self))

    def all_unique[U](self) -> bool:
        """Returns True if all the elements of **self** are unique.

        The function returns as soon as the first non-unique element is encountered.

        Elements are assumed to be hashable.

        If you need to check uniqueness based on a custom key function, use `PyoIterable::all_unique_by` instead.

        Note:
            - On `PyoSequence` and subclasses, this is overriden to directly use an efficient `set` access and length comparison.
            - On `PyoSet`, `PyoMapping` and their subclasses, this directly returns `True`.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Iter, Dict
        >>> Iter("ABCB").all_unique()
        False
        >>> Iter("ABCb").all_unique()
        True
        >>> data = Dict.from_ref({1: "a", 2: "a"})
        >>> data.all_unique()
        True
        >>> data.values().all_unique()
        False

        ```
        """
        return tls.all_unique(iter(self))

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


class PyoCollection[T](PyoIterable[T], Collection[T], ABC):
    """`Extends `PyoIterable[T]` and `collections.abc.Collection[T]`.

    This includes `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, etc...

    Any concrete subclass must implement the required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def length(self) -> int:
        return len(self)

    @override
    def all_unique(self) -> bool:
        return len(self) == len(frozenset(self))

    def contains(self, value: T) -> bool:
        """Check if the `Collection` contains the specified **value**.

        This is equivalent to using the `in` keyword directly on the `Collection`.

        Args:
            value (T): The value to check for existence.

        Returns:
            bool: True if the value exists in the Collection, False otherwise.

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict.from_ref({1: "a", 2: "b"})
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
        >>> from pyochain import Seq
        >>> Seq((1, 2)).repeat(3).collect()
        Seq(Seq(1, 2), Seq(1, 2), Seq(1, 2))
        >>> Seq(("a", "b")).repeat(2).collect()
        Seq(Seq('a', 'b'), Seq('a', 'b'))
        >>> Seq([0]).repeat().flatten().take(5).collect()
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
        >>> from pyochain import Dict
        >>> d = Dict(())
        >>> d.is_empty()
        True
        >>> d.insert(1, "a")
        NONE
        >>> d.is_empty()
        False

        ```
        """
        return len(self) == 0


class PyoIterator[T](PyoIterable[T], Iterator[T], ABC):
    """Extends `PyoIterable[T]` and `collections.abc.Iterator[T]`.

    Is the base class for `Iter[T]`.
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

        Note:
            The actual `.__next__()` method must be conform to the Python `Iterator` Protocol, and is what will be actually called if you iterate over the `PyoIterator` instance.

            `PyoIterator.next()` is a convenience method that wraps the result in an `Option` to handle exhaustion gracefully, for custom use cases.

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
            This is similar to `reduce()` but with an initial value, making it equivalent to
            Python `functools.reduce()` with an initializer.

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
        >>> Iter([]).try_fold(0, checked_add)
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
        >>> Iter([]).try_reduce(checked_add)
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
        >>> Iter([3, 2, 1, 0, 4, 2, 1, 0]).argmin()
        3
        >>> # Example 2: look up a label corresponding to the position of a value that minimizes a cost function
        >>> def cost(x: int) -> float:
        ...     "Days for a wound to heal given a subject's age."
        ...     return x**2 - 20 * x + 150
        >>>
        >>> labels = Seq(["homer", "marge", "bart", "lisa", "maggie"])
        >>> ages = Seq([35, 30, 10, 9, 1])
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
        >>> Iter([Some(1), Some(2), Some(3)]).try_collect()
        Some(Vec(1, 2, 3))
        >>> # Failing to collect in the same way:
        >>> Iter([Some(1), Some(2), NONE, Some(3)]).try_collect()
        NONE
        >>> # A similar example, but with Result:
        >>> Iter([Ok(1), Ok(2), Ok(3)]).try_collect()
        Some(Vec(1, 2, 3))
        >>> Iter([Ok(1), Err("error"), Ok(3)]).try_collect()
        NONE
        >>> def external_fn(x: int) -> Option[int]:
        ...     if x % 2 == 0:
        ...         return Some(x)
        ...     return NONE
        >>> Iter([1, 2, 3, 4]).map(external_fn).try_collect()
        NONE
        >>> # Demonstrating that the iterator remains usable after a failure:
        >>> it = Iter([Some(1), NONE, Some(3), Some(4)])
        >>> it.try_collect()
        NONE
        >>> it.try_collect()
        Some(Vec(3, 4))

        ```
        """
        from .._iter import Vec

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
        from .._iter import Vec

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
        >>> peoples = (Person("Alice", 30), Person("Bob", 25), Person("Charlie", 35))
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
        from .._iter import Vec

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


class PyoSequence[T](PyoCollection[T], Sequence[T], ABC):
    """Extends `PyoCollection[T]` and `collections.abc.Sequence[T]`.

    Is the shared ABC for concrete sequences: `Seq`, `Range` and `Vec`.

    Any concrete subclass must implement the required `Sequence` dunder methods:

    - `__getitem__`
    - `__len__`
    - `__contains__`
    - `__iter__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def first(self) -> T:
        return self[0]

    @override
    def second(self) -> T:
        return self[1]

    @override
    def last(self) -> T:
        return self[-1]

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
        >>> from pyochain import Seq
        >>> data = Seq((10, 20, 30))
        >>> data.get(1)
        Some(20)
        >>> data.get(5)
        NONE

        ```
        """
        try:
            return Some(self[index])  # pyright: ignore[reportReturnType]
        except IndexError:
            return NONE

    def rev(self) -> Iter[T]:
        """Return an `Iterator` with the elements of the `Sequence` in reverse order.

        Returns:
            Iter[T]: An `Iterator` with the elements in reverse order.

        Example:
        ```python
        >>> from pyochain import Seq
        >>> Seq((1, 2, 3)).rev().collect()
        Seq(3, 2, 1)

        ```
        """
        from .._iter import Iter

        return Iter(reversed(self))


class PyoSet[T](PyoCollection[T], AbstractSet[T], ABC):
    """Extends `PyoCollection[T]` and `collections.abc.Set[T]`.

    Is the shared ABC for concrete set-like collections: `Set` and `FrozenSet`.

    Any concrete subclass must implement the required `Set` dunder methods:

    - `__contains__`
    - `__iter__`
    - `__len__`

    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def all_unique(self) -> bool:
        return True

    def is_subset(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of this set are in `other` (including equality).

        Returns `True` if every element in this set is also present in `other`.

        This includes the case where both sets are identical.

        Use `is_subset_strict()` to exclude the equality case.

        Args:
            other (AbstractSet[T]): The set to check containment against.

        Returns:
            bool: `True` if all elements are contained, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Set
        >>> Set((1, 2)).is_subset({1, 2, 3})  # All elements present
        True
        >>> Set((1, 2)).is_subset({1, 2})  # Also True: they're equal
        True
        >>> Set((1, 4)).is_subset({1, 2, 3})  # 4 is not in the other set
        False

        ```
        """
        return self <= other

    def is_subset_strict(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of this set are in `other`, excluding equality.

        Returns `True` if every element in this set is also present in `other`, AND `other` contains at least one element not in this set.

        This is a proper (or strict) subset relation.

        Use `is_subset()` if you want to accept equal sets as well.

        Args:
            other (AbstractSet[T]): The set to check strict containment against.

        Returns:
            bool: `True` if this is a strict subset, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Set
        >>> Set((1, 2)).is_subset_strict({1, 2, 3})  # Proper subset
        True
        >>> Set((1, 2)).is_subset_strict({1, 2})  # Equal, not proper
        False
        >>> Set((1, 4)).is_subset_strict({1, 2, 3})  # 4 not contained
        False

        ```
        """
        return self < other

    def eq(self, other: AbstractSet[T]) -> bool:
        """Test whether this set contains exactly the same elements as `other`.

        Sets are equal if they have the same number of elements and every element in one is present in the other.

        Order is irrelevant for sets.

        This is an explicit method; you can also use the `==` operator directly.

        Args:
            other (AbstractSet[T]): The set to compare with.

        Returns:
            bool: `True` if both sets contain identical elements, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Set
        >>> Set((1, 2)).eq({2, 1})  # Same elements, different order
        True
        >>> Set((1, 2)).eq({1, 2, 3})  # Different number of elements
        False
        >>> Set((1, 2)).eq({1, 2})  # Identical
        True

        ```
        """
        return self == other

    def is_superset(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of `other` are in this set (including equality).

        Returns `True` if this set contains every element from `other`.

        This is the inverse of `is_subset()` -> if A is a subset of B, then B is a superset of A.

        Use `is_superset_strict()` (if available) to exclude equality.

        Args:
            other (AbstractSet[T]): The set to check containment for.

        Returns:
            bool: `True` if all elements from `other` are present, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Set
        >>> Set((1, 2, 3)).is_superset({1, 2})  # Contains all
        True
        >>> Set((1, 2)).is_superset({1, 2})  # Also True: they're equal
        True
        >>> Set((1, 2)).is_superset({1, 2, 3})  # Missing element 3
        False

        ```
        """
        return self >= other

    def is_disjoint(self, other: AbstractSet[T]) -> bool:
        """Test whether this set and `other` have no elements in common.

        Returns `True` if the intersection of the two sets is empty.

        This is the opposite of having any overlap.

        Args:
            other (AbstractSet[T]): The set to compare with.

        Returns:
            bool: `True` if no common elements exist, `False` otherwise.

        Example:
        ```python
        >>> from pyochain import Set
        >>> Set((1, 2)).is_disjoint((3, 4))  # No overlap
        True
        >>> Set((1, 2)).is_disjoint((2, 3))  # Share element 2
        False
        >>> Set((1, 2)).is_disjoint((1, 2))  # Identical sets
        False

        ```
        """
        return self.isdisjoint(other)


class PyoMappingView[T](MappingView, PyoCollection[T], ABC):
    """Extends both `MappingView` from `collections.abc` and `PyoCollection[T]`.

    Is the base class shared by the views returned by `PyoMapping` methods.

    Any concrete subclass must implement the required `MappingView` dunder methods:

    - `__contains__`
    - `__iter__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoMapping[K, V](PyoCollection[K], Mapping[K, V], ABC):
    """Extends `PyoCollection[K]` and `collections.abc.Mapping[K, V]`.

    Serves as a base class for pyochain mappings, such as `Dict`.

    Any concrete subclass must implement the required `Mapping` dunder methods:

    - `__getitem__`
    - `__iter__`
    - `__len__`

    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def all_unique(self) -> bool:
        return True

    @override
    def keys(self) -> PyoKeysView[K]:
        """Return a view of the `Mapping` keys.

        Returns:
            PyoKeysView[K]: A view of the dictionary's keys.

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict({1: "a", 2: "b"})
        >>> data.keys()
        PyoKeysView(Dict(1: 'a', 2: 'b'))

        ```
        """
        from .._set import PyoKeysView

        return PyoKeysView(self)

    @override
    def values(self) -> PyoValuesView[V]:
        """Return a view of the `Mapping` values.

        Returns:
            PyoValuesView[V]: A view of the dictionary's values.

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict({1: "a", 2: "b"})
        >>> data.values()
        PyoValuesView(Dict(1: 'a', 2: 'b'))

        ```
        """
        from .._set import PyoValuesView

        return PyoValuesView(self)

    @override
    def items(self) -> PyoItemsView[K, V]:
        """Return a view of the `Mapping` items.

        Returns:
            PyoItemsView[K, V]: A view of the dictionary's (key, value) pairs.

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict({1: "a", 2: "b"})
        >>> data.items()
        PyoItemsView(Dict(1: 'a', 2: 'b'))

        ```
        """
        from .._set import PyoItemsView

        return PyoItemsView(self)


class PyoMutableMapping[K, V](PyoMapping[K, V], MutableMapping[K, V], ABC):
    """Extends `PyoMapping[K, V]` and `collections.abc.MutableMapping[K, V]`.

    Serves as a base class for pyochain mutable mappings, such as `Dict`.

    Any concrete subclass must implement the required `MutableMapping` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__iter__`
    - `__len__`

    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

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
        >>> from pyochain import Dict
        >>> data = Dict(())
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
        return option(previous)

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
        >>> from pyochain import Dict
        >>> d = Dict(())
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

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict({1: "a", 2: "b"})
        >>> data.remove(1)
        Some('a')
        >>> data.remove(3)
        NONE

        ```
        """
        return option(self.pop(key, None))

    def remove_entry(self, key: K) -> Option[tuple[K, V]]:
        """Remove a key from the `MutableMapping` and return the item if it existed.

        Return an `Option[tuple[K, V]]` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[tuple[K, V]]: `Some((key, value))` pair associated with the removed key, or `None` if the **key** was not present.

        Example:
        ```python
        >>> from pyochain import Dict
        >>> data = Dict({1: "a", 2: "b"})
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
        >>> from pyochain import Dict
        >>> data = Dict.from_ref({"a": 1})
        >>> data.get_item("a")
        Some(1)
        >>> data.get_item("x").unwrap_or("Not Found")
        'Not Found'

        ```
        """
        return option(self.get(key, None))


class PyoMutableSequence[T](PyoSequence[T], MutableSequence[T], ABC):
    """Extends `PyoSequence[T]` and `collections.abc.MutableSequence[T]`.

    This ABC is the base class for mutable sequence types in pyochain, such as `Vec`.

    Any concrete subclass must implement the required `MutableSequence` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__len__`
    - `insert`

    This class notably provides various methods inspired from Rust's `Vec` type, which provides memory-efficient in-place operations.

    They are slower than simple `.extend()`, slices and `clear()` calls, but avoids all intermediate allocations, making them suitable for large collections where memory usage is a concern.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def retain(self, predicate: Callable[[T], bool]) -> None:
        """Retains only the elements specified by the *predicate*.

        In other words, remove all elements e for which the *predicate* function returns `False`.

        This method operates in place, visiting each element exactly once in forward order, and preserves the order of the retained elements.

        Note:
            This is similar to filtering, but operates in place without allocating a new collection once collected.

            For example `new_list = list(filter(predicate, my_list))` followed by `my_list.clear()` would allocate a new collection before clearing the original, resulting in higher peak memory usage.

        Args:
            predicate (Callable[[T], bool]): A function that returns `True` for elements to keep and `False` for elements to remove.

        Example:
        ```python
        >>> from pyochain import Vec, Seq
        >>> vec = Vec((1, 2, 3, 4))
        >>> vec.retain(lambda x: x % 2 == 0)
        >>> vec
        Vec(2, 4)

        ```
        External state may be used to decide which elements to keep.

        ```python
        >>> vec = Vec((1, 2, 3, 4, 5))
        >>> keep = Seq((False, True, True, False, True)).iter()
        >>> vec.retain(lambda _: next(keep))
        >>> vec
        Vec(2, 3, 5)

        ```
        """
        return tls.retain(self, predicate)

    # NOTE: need to check what does `pop` do really here regarding index
    def truncate(self, length: int) -> None:
        """Shortens the `MutableSequence`, keeping the first *length* elements and dropping the rest.

        If *length* is greater or equal to the `MutableSequence` current `__len__()`, this has no effect.

        The `Vec::drain` method can emulate `Vec::truncate`, but causes the excess elements to be returned instead of dropped.

        Note:
            This is equivalent to `del seq[length:]`, except that it won't create an intermediate slice object.

        Args:
            length (int): The length to truncate the `MutableSequence` to.

        Example:
        ```python
        >>> from pyochain import Vec
        >>> # Truncating a five element vector to two elements:
        >>> vec = Vec((1, 2, 3, 4, 5))
        >>> vec.truncate(2)
        >>> vec
        Vec(1, 2)

        ```
        No truncation occurs when len is greater than the `MutableSequence` current length:
        ```python
        >>> from pyochain import Vec
        >>> vec = Vec((1, 2, 3))
        >>> vec.truncate(8)
        >>> vec
        Vec(1, 2, 3)

        ```
        Truncating when len == 0 is equivalent to calling the clear method.
        ```python
        >>> from pyochain import Vec
        >>> vec = Vec((1, 2, 3))
        >>> vec.truncate(0)
        >>> vec
        Vec()

        ```
        """
        pop = self.pop
        for _ in range(len(self) - length):
            _ = pop()

    # NOTE: Rust does not support MutableSequence ATM. We either need to find a new implementation, or wait until MutableSequence is supported to implement this method.
    def extend_move(self, other: Self | list[T]) -> None:
        """Moves all the elements of *other* into *self*, leaving *other* empty.

        This is equivalent to `extend(other)` followed by `other.clear()`, but avoids intermediate allocations by moving elements one at a time.

        Each element is extracted from **other**, appended to **self**, and removed from **other** in a single step.

        Args:
            other (Self | list[T]): The other `MutableSequence` to move elements from.

        Example:
        ```python
        >>> from pyochain import Vec
        >>> v1 = Vec((1, 2, 3))
        >>> v2 = Vec((4, 5, 6))
        >>> v1.extend_move(v2)
        >>> v1
        Vec(1, 2, 3, 4, 5, 6)
        >>> v2
        Vec()

        ```
        If we compare to extend

        ```python
        >>> v1 = Vec((1, 2, 3))
        >>> v2 = Vec((4, 5, 6))
        >>> v1.extend(v2)
        >>> v1
        Vec(1, 2, 3, 4, 5, 6)
        >>> # At this point v2 is still intact,
        >>> # meaning that we have a full intermediate copy of v2 in memory,
        >>> # which is not the case with extend_move
        >>> v2
        Vec(4, 5, 6)
        >>> v2.clear()
        >>> v2
        Vec()

        ```
        """
        pop = other.pop
        self.extend(pop(0) for _ in range(len(other)))

    def extract_if(
        self, predicate: Callable[[T], bool], start: int = 0, end: int | None = None
    ) -> Iter[T]:
        """Creates an `Iter` which uses a *predicate* to determine if an element in `Self` should be removed.

        If the *predicate* returns `True`, the element is removed from `Self` and yielded.

        If the *predicate* returns `False`, the element remains in `Self` and will not be yielded.

        You can specify a range for the extraction.

        If the returned `Iterator` is not exhausted, e.g. because it is dropped without iterating or the iteration short-circuits, then the remaining elements will be retained.

        Using this method is equivalent to the following code:
        ```python
        data = Vec((...))
        for i in range(data.length()):
            if predicate(data[i]):
                val = data.pop(i)
                # your code here
        ```

        Args:
            predicate (Callable[[T], bool]): A function that takes an element and returns `True` if it should be extracted, or `False` if it should be retained.
            start (int): The starting index of the range to consider for extraction. Defaults to `0`.
            end (int | None): The ending index of the range to consider for extraction. Defaults to `None`, which means the end of `Self`.

        Returns:
            Iter[T]: An `Iter` that yields the extracted elements.

        Example:
        ```python
        >>> from pyochain import Vec
        >>> data = (1, 2, 3, 4, 5)
        >>> vec = Vec(data)
        >>> extracted = vec.extract_if(lambda x: x % 2 == 0).collect()
        >>> extracted
        Seq(2, 4)
        >>> vec
        Vec(1, 3, 5)
        >>> # Extracting with a range
        >>> vec = Vec(data)
        >>> extracted = vec.extract_if(lambda x: x % 2 == 0, start=1, end=4).collect()
        >>> extracted
        Seq(2, 4)
        >>> vec
        Vec(1, 3, 5)

        ```
        """
        from .._iter import Iter

        def _extract_if_gen() -> Iterator[T]:
            effective_end = end if end is not None else len(self)
            i = start
            pop = self.pop
            while i < effective_end and i < len(self):
                if predicate(self[i]):
                    yield pop(i)
                    effective_end -= 1
                else:
                    i += 1

        return Iter(_extract_if_gen())

    def drain(self, start: int | None = None, end: int | None = None) -> Iter[T]:
        """Removes the subslice indicated by the given *start* and *end* from the `Vec`, returning an `Iterator` over the removed subslice.

        If the `Iterator` is dropped before being fully consumed, it drops the remaining removed elements.

        Args:
            start (int | None): Starting index of the subslice to drain. Defaults to `0` if `None`.
            end (int | None): Ending index of the subslice to drain. Defaults to `len(self)` if `None`.

        Returns:
            Iter[T]: An `Iterator` over the drained elements.

        Example:
        ```python
        >>> from pyochain import Vec
        >>> v = Vec.from_ref([1, 2, 3])
        >>> u = v.drain(1).collect()
        >>> v
        Vec(1)
        >>> u
        Seq(2, 3)
        >>> # A full range clears the vector, like `clear()` does
        >>> _ = v.drain().collect()
        >>> v
        Vec()

        ```
        Fully consuming the `Iterator` removes all drained elements
        ```python
        >>> from pyochain import Vec
        >>> v = Vec.from_ref([1, 2, 3])
        >>> _ = v.drain(0, 3).collect()
        >>> v
        Vec()

        ```
        """
        from .._iter import Iter

        return Iter(DrainIterator(self, start or 0, end or len(self)))
