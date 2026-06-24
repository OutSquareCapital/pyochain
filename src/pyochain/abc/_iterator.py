from __future__ import annotations

import functools
import itertools
from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, Self, overload, override

from .._abc import (  # pyright: ignore[reportMissingModuleSource]
    PyoIterable,
    PyoIteratorRS,
)
from ..rs import NONE, Err, Null, Ok, Option, Result, Some, option

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Generator,
        Iterable,
        Iterator,
        KeysView,
        MutableSequence,
        Sequence,
        ValuesView,
    )

    from .._dict import Dict
    from .._iter import Iter
    from .._range import Range
    from .._seq import Seq
    from .._set import Set, SetMut
    from .._types import (
        LiteralInteger,
        SupportsAnyAdd,
        SupportsAnyRichComparison,
        SupportsComparison,
        SupportsSumWithNoDefaultGiven,
    )
    from .._vec import Vec
    from ..collections import Deque
    from ._sequences import PyoMutableSequence

    type AnyIter = Iterable[Any]  # pyright: ignore[reportExplicitAny]
    type SupportsAnyComparison = SupportsComparison[Any]  # pyright: ignore[reportExplicitAny]


class PyoIterator[T](PyoIteratorRS[T], ABC):
    """Extends `PyoIterable[T]` and `collections.abc.Iterator[T]`.

    - An `Iterable` is any object capable of creating an `Iterator` (i.e., it implements the `__iter__()` method).
    - An `Iterator` is an object representing a stream of data, generating the next value with each call to `__next__()`.

    `Iterator`s are composable, meaning you can chain operations like `map()`, `filter()`, etc., that will simply add a new step to the processing pipeline without executing it.

    Thus, it can be considered akin to a SQL query: An `Iterator` represents a recipe for how to process the data.

    Terminal operations (like `collect()`, `count()`, `all()`, etc.) will "execute the query" by consuming the `Iterator` and producing a final result.

    This is done by calling `__next__()` repeatedly until `StopIteration` is raised, which signals that the `Iterator` is exhausted.

    Once this happened, the `Iterator` instance is empty and cannot be reused to produce new values.

    A high-level way of thinking about how to use `Iterators` is to create one from a source of data, build a plan, and execute it.

    Then, if the result is a new `Iterable`, you can create a new `Iterator` from it and repeat the process.

    If all of this doesn't sound familiar, it's simply because Python does this in an implicit way.

    A *for loop* will create an `Iterator` from the provided iterable, and consume it until exhaustion.

    For example, a `list` knows its size, how to access items by index, etc..

    But it does not know how to iterate over itself, i.e returns elements one by one and stop once x event happens.

    It knows, however, how to create an `Iterator` object that will handle this.

    All concrete subclasses must implement the required `Iterator` dunder methods:

    - `__iter__`
    - `__next__`

    Example:
        ```python
        >>> from pyochain import Seq
        >>> from pyochain.abc import PyoIterator
        >>>
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
        >>> counter.iter().take(3).collect(Seq)
        Seq(7, 8, 9)

        ```
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @classmethod
    def once[V](cls, value: V) -> PyoIterator[V]:
        """Create an `Iterator` that yields a single value.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.

        This can be considered the equivalent of `.insert()` but as a constructor.

        Args:
            value (V): The single value to yield.

        Returns:
            PyoIterator[V]: An `Iterator` yielding the specified value.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter.once(42).collect(Seq)
            Seq(42,)

            ```
        """
        return cls._from_iterable((value,))

    @classmethod
    def once_with[**P, R](
        cls, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> PyoIterator[R]:
        """Create an `Iterator`  that lazily generates a value exactly once by invoking the provided closure.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.

        This can be considered the equivalent of [`PyoIterator::insert`][PyoIterator.insert] but as a constructor.

        Unlike `PyoIterator::once`, this function will lazily generate the value on request.

        Args:
            func (Callable[P, R]): The single value to yield.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            PyoIterator[R]: An `Iterator` yielding the specified value.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter.once_with(lambda: 42).collect(Seq)
            Seq(42,)

            ```
        """

        def _once_with() -> Generator[R]:
            yield func(*args, **kwargs)

        return cls._from_iterable(_once_with())

    @classmethod
    def from_repeat[O](cls, obj: O, n: int | None = None) -> PyoIterator[O]:
        """Repeat the provided object **n** times (as elements) as elements of an `Iterator`.

        If **n** is `None`, this will create an infinite `Iterator`.

        Be sure to use [`PyoIterator::take`][PyoIterator.take] or [`PyoIterator::slice`][PyoIterator.slice] to limit the number of items taken.

        Warning:
            Each repetition is a reference to the same object, not a copy.

            This means that if the object is mutable and you modify one of the repetitions, all next repetitions will reflect that change.

        Args:
            obj (O): The object to repeat.
            n (int | None): Optional number of repetitions.

        Returns:
            PyoIterator[O]: An `Iterator` of repeated **obj**.

        See Also:
            [`PyoIterator::cycle`][cycle] to repeat the **elements** of the `Iterator`.
            [`PyoIterator::repeat`][repeat] to repeat the **entire** `Iterator`.

        Example:
            ```python
            >>> from pyochain import Seq, Iter
            >>> Iter.from_repeat(1, 3).collect(Seq)
            Seq(1, 1, 1)
            >>> Iter.from_repeat(("a", "b"), 2).collect(Seq)
            Seq(('a', 'b'), ('a', 'b'))

            ```
            Shared reference behavior:
            ```python
            >>> from pyochain import Vec
            >>>
            >>> base = ["Alice", "Bob", "Charlie"]
            >>>
            >>> first, second = Iter.from_repeat(base).take(2).collect(tuple)
            >>> first.append("Joe")
            >>> first
            ['Alice', 'Bob', 'Charlie', 'Joe']
            >>> base
            ['Alice', 'Bob', 'Charlie', 'Joe']
            >>> second
            ['Alice', 'Bob', 'Charlie', 'Joe']
            >>> first is second and first is base and second is base
            True

            ```
        """
        match n:
            case None:
                return cls._from_iterable(itertools.repeat(obj))
            case _:
                return cls._from_iterable(itertools.repeat(obj, n))

    def nth(self, n: int) -> Option[T]:
        """Return the nth item of the `Iterable` at the specified *n*.

        This is similar to `__getitem__` but for lazy `Iterators`.

        If *n* is out of bounds, returns `NONE`.

        Args:
            n (int): The index of the item to retrieve. It must be a non-negative integer.

        Returns:
            Option[T]: `Some(item)` at the specified *n*.

        Example:
            ```python
            >>> from pyochain import Range
            >>> data = Range(0, 10)
            >>> data.iter().nth(1)
            Some(1)
            >>> data.iter().nth(10)
            NONE

            ```
        """
        return option(next(itertools.islice(iter(self), n, n + 1), None))

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
        return functools.reduce(func, iter(self))

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
        return functools.reduce(func, iter(self), init)

    def collect[R: Collection[Any]](self, collector: Callable[[Iterator[T]], R]) -> R:
        """Transforms the `Iterator` into a collection.

        The most basic pattern in which `collect()` is used is to turn one collection into another.

        You take a collection, call `iter()` on it, do a bunch of transformations, and then `collect()` at the end.

        You specify the target `Collection` type by providing a **collector** function or type.

        This can be any `Callable` that takes an `Iterator[T]` and returns a `Collection[T]` of those types.

        This is equivalent to `Pipe::pipe` at runtime, but with a few differences:

            - A narrower constraint (`Collection[Any]`) to specify the intent
            - Better performance (no args/kwargs unpacking).

        If you need to pass additional arguments, you can use [`Pipe::pipe`][Pipe.pipe] instead.

        Args:
            collector (Callable[[Iterator[T]], R]): Function|type that defines the target collection.

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

    def sort[U: SupportsAnyRichComparison](
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
        self, key: Callable[[T], SupportsAnyRichComparison], *, reverse: bool = False
    ) -> Vec[T]:
        """Sort the elements of the sequence transformed by the key function.

        Note:
            This method must consume the entire `Iterator` to perform the sort.

            The result is a new `Vec` over the sorted sequence.

        Args:
            key (Callable[[T], SupportsAnyRichComparison]): Function to extract a comparison key from each element.
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[T]: A `Vec` with elements sorted.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> str_numbers = Seq(("3", "1", "2"))
            >>> str_numbers.iter().sort_by(int)
            Vec('1', '2', '3')
            >>> str_numbers.iter().sort_by(int, reverse=True)
            Vec('3', '2', '1')
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            >>>
            >>> peoples = Seq((
            ...     Person("Alice", 30),
            ...     Person("Bob", 25),
            ...     Person("Charlie", 35),
            ... ))
            >>> sorted_names = (
            ...     peoples
            ...     .iter()
            ...     .sort_by(lambda x: x.age)
            ...     .iter()
            ...     .map(lambda x: x.name)
            ...     .collect(Seq)
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

    @overload
    def sum(self: PyoIterator[bool], start: int = 0) -> int: ...
    @overload
    def sum(self: PyoIterator[LiteralInteger], start: int = 0) -> int: ...
    @overload
    def sum[T1: SupportsSumWithNoDefaultGiven](
        self: PyoIterator[T1],
    ) -> T1 | Literal[0]: ...
    @overload
    def sum[A1: SupportsAnyAdd, A2: SupportsAnyAdd](
        self: PyoIterator[A1], start: A2
    ) -> A1 | A2: ...
    def sum[T1: SupportsSumWithNoDefaultGiven, A1: SupportsAnyAdd, A2: SupportsAnyAdd](
        self: PyoIterator[bool | LiteralInteger] | PyoIterator[T1] | PyoIterator[A1],
        start: int | T1 | A2 = 0,
    ) -> int | T1 | A1 | A2:
        """Return the sum of the `Iterator`.

        If the `Iterator` is empty (i.e., yields no elements), return the value of `start` (which defaults to `0`).

        Args:
            start (int | T1 | A2): The value to return if the `Iterator` is empty.

        Returns:
            int | T1 | A1 | A2: The sum of all elements.

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

    def min[U: SupportsAnyRichComparison](self: PyoIterable[U]) -> U:
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

    def min_by[U: SupportsAnyRichComparison](self, key: Callable[[T], U]) -> T:
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

    def max[U: SupportsAnyRichComparison](self: PyoIterable[U]) -> U:
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

    def max_by[U: SupportsAnyRichComparison](self, key: Callable[[T], U]) -> T:
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

    def take(self, n: int) -> PyoIterator[T]:
        """Creates an iterator that yields the first n elements, or fewer if the underlying iterator ends sooner.

        `Iter.take(n)` yields elements until n elements are yielded or the end of the iterator is reached (whichever happens first).

        The returned iterator is either:

        - A prefix of length n if the original iterator contains at least n elements
        - All of the (fewer than n) elements of the original iterator if it contains fewer than n elements.

        Args:
            n (int): Number of elements to take.

        Returns:
            PyoIterator[T]: An `Iterator` of the first n items.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((1, 2, 3))
            >>> data.iter().take(2).collect(Seq)
            Seq(1, 2)
            >>> data.iter().take(5).collect(Seq)
            Seq(1, 2, 3)

            ```
        """
        return self._from_iterable(itertools.islice(iter(self), n))

    def skip(self, n: int) -> PyoIterator[T]:
        """Create an `Iterator` that skips the first n elements.

        skip(**n**) skips elements until n elements are skipped or the end of the `Iterator` is reached (whichever happens first).

        After that, all the remaining elements are yielded.

        In particular, if the original `Iterator` is too short, then the returned `Iterator` is empty.

        If **n** is negative or zero, the original `Iterator` is returned unchanged.

        Args:
            n (int): Number of elements to skip.

        Returns:
            PyoIterator[T]: An `Iterator` of the remaining elements.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((1, 2, 3))
            >>> data.iter().skip(1).collect(Seq)
            Seq(2, 3)
            >>> data.iter().skip(5).collect(Seq)
            Seq()
            >>> data.iter().skip(0).collect(Seq)
            Seq(1, 2, 3)

            ```
        """
        return self._from_iterable(itertools.islice(iter(self), n, None))

    def step_by(self, step: int) -> PyoIterator[T]:
        """Creates an `Iterator` starting at the same point, but stepping by the given **step** at each iteration.

        Note:
            The first element of the iterator will always be returned, regardless of the **step** given.

        Args:
            step (int): Step size for selecting items.

        Returns:
            PyoIterator[T]: An `Iterator` of every nth item.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((0, 1, 2, 3, 4, 5)).iter().step_by(2).collect(Seq)
            Seq(0, 2, 4)

            ```
        """
        return self._from_iterable(itertools.islice(iter(self), 0, None, step))

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> PyoIterator[T]:
        """Return a slice of the `Iterator`.

        Args:
            start (int | None): Starting index of the slice.
            stop (int | None): Ending index of the slice.
            step (int | None): Step size for the slice.

        Returns:
            PyoIterator[T]: An `Iterator` of the sliced items.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((1, 2, 3, 4, 5))
            >>> data.iter().slice(1, 4).collect(Seq)
            Seq(2, 3, 4)
            >>> data.iter().slice(step=2).collect(Seq)
            Seq(1, 3, 5)

            ```
        """
        return self._from_iterable(itertools.islice(iter(self), start, stop, step))

    def insert(self, value: T) -> PyoIterator[T]:
        """Prepend the *value* to the `Iterator`.

        Note:
            This can be considered the equivalent as `list.append()`, but for a lazy `Iterator`.

            However, append add the value at the **end**, while insert add it at the **beginning**.

        See Also:
            [`PyoIterator::chain`][chain] to add multiple elements at the end of the `Iterator`.

        Args:
            value (T): The value to prepend.

        Returns:
            PyoIterator[T]: A new Iterable wrapper with the value prepended.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((2, 3)).insert(1).collect(Seq)
            Seq(1, 2, 3)

            ```
        """
        return self._from_iterable(itertools.chain((value,), iter(self)))

    def chain(self, *others: Iterable[T]) -> PyoIterator[T]:
        """Concatenate **self** with one or more `Iterables`, any of which may be infinite.

        In other words, it links **self** and **others** together, in a chain. 🔗

        An infinite `Iterable` will prevent the rest of the arguments from being included.

        This is equivalent to `list.extend()`, except it is fully lazy and works with any `Iterable`.

        See Also:
            [`PyoIterator::insert`][insert] to add a single element at the beginning of the `Iterator`.

        Args:
            *others (Iterable[T]): Other iterables to concatenate.

        Returns:
            PyoIterator[T]: A new `Iterator` which will first iterate over values from the original `Iterator` and then over values from the **others** `Iterable`s.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2)).chain((3, 4), [5]).collect(Seq)
            Seq(1, 2, 3, 4, 5)
            >>> Iter((1, 2)).chain(Iter.from_count(3)).take(5).collect(Seq)
            Seq(1, 2, 3, 4, 5)

            ```
        """
        return self._from_iterable(itertools.chain.from_iterable((iter(self), *others)))

    def peekable(self) -> Peekable[T]:
        """Creates an iterator which can use the peek and peek_mut methods to look at the next element of the `Iterator` without consuming it.

        See their documentation for more information.

        Note that the underlying `Iterator` is still advanced when peek or peek_mut are called for the first time.

        In order to retrieve the next element, `next` is called on the underlying `Iterator`, hence any side effects (i.e. anything other than fetching the next value) of the `next` method will occur.

        Returns:
            Peekable[T]: A new `Iterator` that allows peeking at the next element.

        Examples:
            Basic usage:
            ```python
            >>> from pyochain import Range
            >>> xs = Range(1, 4)
            >>> iterator = xs.iter().peekable()
            >>> # peek() lets us see into the future
            >>> iterator.peek()
            Some(1)
            >>> iterator.next()
            Some(1)
            >>> iterator.next()
            Some(2)
            >>> # we can peek() multiple times, the iterator won't advance
            >>> iterator.peek()
            Some(3)
            >>> iterator.peek()
            Some(3)
            >>> iterator.next()
            Some(3)
            >>> # after the iterator is finished, so is peek()
            >>> iterator.peek()
            NONE
            >>> iterator.next()
            NONE

        """
        return Peekable(iter(self))

    @overload
    def flatten[U](self: PyoIterator[KeysView[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Iterable[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Generator[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[ValuesView[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Iterator[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Collection[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Sequence[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[list[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[tuple[U, ...]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[PyoIterator[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Iter[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Seq[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Set[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[SetMut[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten[U](self: PyoIterator[Vec[U]]) -> PyoIterator[U]: ...
    @overload
    def flatten(self: PyoIterator[range]) -> PyoIterator[int]: ...
    @overload
    def flatten(self: PyoIterator[Range]) -> PyoIterator[int]: ...
    @overload
    def flatten[U](self: PyoIterator[Dict[U, Any]]) -> PyoIterator[U]: ...  # pyright: ignore[reportExplicitAny]
    def flatten[U: AnyIter](self: PyoIterator[U]) -> PyoIterator[Any]:  # pyright: ignore[reportExplicitAny]
        """Creates an `Iterator` that flattens nested structures.

        This is useful when you have an `Iterator` of `Iterable` and you want to remove one level of indirection.

        Returns:
            PyoIterator[Any]: An `Iterator` of flattened elements.

        Example:
            Basic usage:
            ```python
            >>> from pyochain import Iter, Seq
            >>> data = ((1, 2, 3, 4), (5, 6))
            >>> flattened = Iter(data).flatten().collect(Seq)
            >>> flattened
            Seq(1, 2, 3, 4, 5, 6)

            ```
            Mapping and then flattening:
            ```python
            >>> from pyochain import Seq
            >>> words = Seq(("alpha", "beta", "gamma"))
            >>> merged = words.iter().flatten().collect(Seq)
            >>> merged
            Seq('a', 'l', 'p', 'h', 'a', 'b', 'e', 't', 'a', 'g', 'a', 'm', 'm', 'a')

            ```
            Flattening only removes one level of nesting at a time:
            ```python
            >>> from pyochain import Seq
            >>> d3 = Seq((((1, 2), (3, 4)), ((5, 6), (7, 8))))
            >>> d2 = d3.iter().flatten().collect(Seq)
            >>> d2
            Seq((1, 2), (3, 4), (5, 6), (7, 8))
            >>> d1 = d3.iter().flatten().flatten().collect(Seq)
            >>> d1
            Seq(1, 2, 3, 4, 5, 6, 7, 8)

            ```
            Here we see that `flatten()` does not perform a “deep” flatten.

            Instead, only **one** level of nesting is removed.

            That is, if you `flatten()` a three-dimensional array, the result will be two-dimensional and not one-dimensional.

            To get a one-dimensional structure, you have to `flatten()` again.

        """
        return self._from_iterable(itertools.chain.from_iterable(iter(self)))

    def flat_map[R](self, func: Callable[[T], Iterable[R]]) -> PyoIterator[R]:
        """Creates an iterator that applies a function to each element of the original iterator and flattens the result.

        This is useful when the **func** you want to pass to `.map()` itself returns an iterable, and you want to avoid having nested iterables in the output.

        This is equivalent to calling `.map(func).flatten()`.

        Args:
            func (Callable[[T], Iterable[R]]): Function to apply to each element.

        Returns:
            PyoIterator[R]: An iterable of flattened transformed elements.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).flat_map(lambda x: range(x)).collect(Seq)
            Seq(0, 0, 1, 0, 1, 2)

            ```
        """
        return self._from_iterable(itertools.chain.from_iterable(map(func, iter(self))))

    def find_map[R](self, func: Callable[[T], Option[R]]) -> Option[R]:
        """Applies function to the elements of the `Iterator` and returns the first Some(R) result.

        `Iter.find_map(f)` is equivalent to `Iter.filter_map(f).next()`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element, returning an `Option[R]`.

        Returns:
            Option[R]: The first `Some(R)` result from applying `func`, or `NONE` if no such result is found.

        Example:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> def _parse(s: str) -> Option[int]:
            ...     try:
            ...         return Some(int(s))
            ...     except ValueError:
            ...         return NONE
            >>>
            >>> Iter(["lol", "NaN", "2", "5"]).find_map(_parse)
            Some(2)

            ```
        """
        return self.filter_map(func).next()

    @overload
    def map_with[T1, R](
        self, iterable: Iterable[T1], /, *, func: Callable[[T, T1], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        func: Callable[[T, T1, T2], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        func: Callable[[T, T1, T2, T3], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
        *,
        func: Callable[[T, T1, T2, T3, T4], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, T5, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
        *,
        func: Callable[[T, T1, T2, T3, T4, T5], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[R](
        self,
        iterable: AnyIter,
        iter2: AnyIter,
        iter3: AnyIter,
        iter4: AnyIter,
        iter5: AnyIter,
        iter6: AnyIter,
        /,
        *iterables: AnyIter,
        func: Callable[..., R],
    ) -> PyoIterator[R]: ...
    def map_with[R](
        self, *iterables: AnyIter, func: Callable[..., R]
    ) -> PyoIterator[R]:
        """Applies a function to the elements of this `Iterator` and additional iterables.

        The provided function must take as many arguments as the number of iterables provided (including **self**).

        It is then applied to the items from all iterables in parallel.

        the iterator stops when the shortest iterable is exhausted.

        Args:
            *iterables (AnyIter): Additional iterables to zip with **self**.
            func (Callable[..., R]): Function to apply to the elements of the iterables.

        Returns:
            PyoIterator[R]: An `Iterator` of results from applying the function to the elements of the iterables.

        See Also:
            [`PyoIterator::map_juxt`][map_juxt] to apply multiple functions to the same elements of the `Iterator`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Triangle:
            ...     x: int
            ...     y: int
            ...     z: int
            >>>
            >>> x = Seq((1, 2, 3))
            >>> y = [4, 5, 6]
            >>> z = [7, 8, 9]
            >>> output = x.iter().map_with(y, z, func=Triangle).collect(Seq)
            >>> output
            Seq(Triangle(x=1, y=4, z=7), Triangle(x=2, y=5, z=8), Triangle(x=3, y=6, z=9))

            ```
        """
        return self._from_iterable(map(func, iter(self), *iterables))

    def repeat(self, n: int | None = None) -> PyoIterator[PyoIterator[T]]:
        """Repeat the entire `Iterator` **n** times (as elements).

        If **n** is `None`, repeat indefinitely.

        Operates lazily, hence if you need to get the underlying elements, you will need to collect each repeated `Iterator` via `.map(lambda x: x.collect(Seq))` or similar.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `PyoIterator::take` or `PyoIterator::slice` to limit the number of items taken.

        See Also:
            [`PyoIterator::cycle`][cycle] to repeat the *elements* of the `PyoIterator` indefinitely.

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            PyoIterator[PyoIterator[T]]: An `Iterator` of repeated `Iterator`s.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>>
            >>> Iter((1, 2)).repeat(3).map(list).collect(Seq)
            Seq([1, 2], [1, 2], [1, 2])

            ```
        """
        new = self._from_iterable

        def repeat(iterator: Iterator[T], n: int | None) -> Iterator[PyoIterator[T]]:

            def _repeat_infinite(iterator: Iterator[T]) -> Generator[PyoIterator[T]]:
                def tee() -> tuple[Iterator[T], ...]:
                    return itertools.tee(iterator, 1)

                iterators = tee()
                while True:
                    yield new(iterators[0])
                    iterators = tee()

            match n:
                case None:
                    return _repeat_infinite(iterator)
                case _:
                    return map(new, itertools.tee(iterator, n))

        return new(repeat(iter(self), n))

    @overload
    def zip[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        strict: bool = ...,
    ) -> PyoIterator[tuple[T, T1]]: ...
    @overload
    def zip[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        strict: bool = ...,
    ) -> PyoIterator[tuple[T, T1, T2]]: ...
    @overload
    def zip[T1, T2, T3](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        strict: bool = ...,
    ) -> PyoIterator[tuple[T, T1, T2, T3]]: ...
    @overload
    def zip[T1, T2, T3, T4](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
        *,
        strict: bool = ...,
    ) -> PyoIterator[tuple[T, T1, T2, T3, T4]]: ...
    def zip(
        self, *others: AnyIter, strict: bool = False
    ) -> PyoIterator[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """Yields n-length tuples, where n is the number of iterables passed as positional arguments.

        The i-th element in every tuple comes from the i-th iterable argument to `.zip()`.

        This continues until the shortest argument is exhausted.

        Note:
            `Iter.map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            *others (AnyIter): Other iterables to zip with.
            strict (bool): If `True` and one of the arguments is exhausted before the others, raise a ValueError.

        Returns:
            PyoIterator[tuple[Any, ...]]: An `Iterator` of tuples containing elements from the zipped `PyoIterator` and other iterables.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>>
            >>> Iter((1, 2)).zip((10, 20)).collect(Seq)
            Seq((1, 10), (2, 20))
            >>> Iter(("a", "b")).zip((1, 2, 3)).collect(Seq)
            Seq(('a', 1), ('b', 2))

            ```
        """
        return self._from_iterable(zip(iter(self), *others, strict=strict))

    @overload
    def product(self, /) -> PyoIterator[tuple[T]]: ...
    @overload
    def product[T2](self, iter2: Iterable[T2], /) -> PyoIterator[tuple[T, T2]]: ...
    @overload
    def product[T2, T3](
        self, iter2: Iterable[T2], iter3: Iterable[T3], /
    ) -> PyoIterator[tuple[T, T2, T3]]: ...
    @overload
    def product[T2, T3, T4](
        self, iter2: Iterable[T2], iter3: Iterable[T3], iter4: Iterable[T4], /
    ) -> PyoIterator[tuple[T, T2, T3, T4]]: ...
    @overload
    def product[T2, T3, T4, T5](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5]]: ...
    @overload
    def product[T2, T3, T4, T5, T6](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8, T9](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        iter9: Iterable[T9],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8, T9]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8, T9, T10](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        iter9: Iterable[T9],
        iter10: Iterable[T10],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8, T9, T10]]: ...
    @overload
    def product(
        self, *iterables: Iterable[T], repeat: int = ...
    ) -> PyoIterator[tuple[T, ...]]: ...
    def product(
        self, *iterables: AnyIter, repeat: int = 1
    ) -> PyoIterator[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """Computes the Cartesian product with other iterables.

        This is the declarative equivalent of nested for-loops.

        It pairs every element from the source iterable with every element from the
        other iterables.

        Args:
            *iterables (AnyIter): Other iterables to compute the Cartesian product with.
            repeat (int): The number of repetitions of the Cartesian product.

        Returns:
            PyoIterator[tuple[Any, ...]]: An iterable of tuples containing elements from the Cartesian product.

        Example:
            ```python
            >>> from pyochain import Seq, Range, Iter
            >>>
            >>> colors = Seq(("blue", "red"))
            >>> sizes = Seq(("S", "M"))
            >>> colors.iter().product(sizes).collect(Seq)
            Seq(('blue', 'S'), ('blue', 'M'), ('red', 'S'), ('red', 'M'))
            >>> res = (
            ...     colors
            ...     .iter()
            ...     .product(sizes)
            ...     .map_star(lambda color, size: f"{color}-{size}")
            ...     .collect(Seq)
            ... )
            >>> res
            Seq('blue-S', 'blue-M', 'red-S', 'red-M')
            >>> res = (
            ...     Range(1, 4)
            ...     .iter()
            ...     .product((10, 20))
            ...     .filter_star(lambda a, b: a * b >= 40)
            ...     .collect(Seq)
            ... )
            >>> res
            Seq((2, 20), (3, 20))
            >>> res = (
            ...     Seq((26, 33))
            ...     .iter()
            ...     .product(("Michael", "Sophie"), ["Engineer"])
            ...     .map_star(lambda age, name, profession: f"{name} is {age} and is {profession}")
            ...     .collect(tuple)
            ... )
            >>> res
            ('Michael is 26 and is Engineer', 'Sophie is 26 and is Engineer', 'Michael is 33 and is Engineer', 'Sophie is 33 and is Engineer')

            ```
            If repeat is specified, the Cartesian product is repeated that many times.
            ```python
            >>> from pyochain import Seq
            >>> colors = Seq(("blue", "red"))
            >>> colors.iter().product(repeat=2).collect(Seq)
            Seq(('blue', 'blue'), ('blue', 'red'), ('red', 'blue'), ('red', 'red'))

            ```
        """
        return self._from_iterable(
            itertools.product(iter(self), *iterables, repeat=repeat)
        )

    def enumerate(self, start: int = 0) -> PyoIterator[tuple[int, T]]:
        """Return a `Iterator` of (index, value) pairs.

        Each value in the `Iterator` is paired with its index, starting from 0.

        Tip:
            `PyoIterator::map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            start (int): The starting index.

        Returns:
            PyoIterator[tuple[int, T]]: An `Iterator` of (index, value) pairs.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> data = ("apple", "banana", "cherry")
            >>> output = Iter(data).enumerate().collect(Seq)
            >>> output
            Seq((0, 'apple'), (1, 'banana'), (2, 'cherry'))
            >>> output = (
            ...     Iter(data)
            ...     .enumerate()
            ...     .map_star(lambda idx, val: (idx, val.upper()))
            ...     .collect(Seq)
            ... )
            >>> output
            Seq((0, 'APPLE'), (1, 'BANANA'), (2, 'CHERRY'))

            ```
        """
        return self._from_iterable(enumerate(iter(self), start))


class Peekable[T](PyoIterator[T]):
    __slots__ = ("_it", "_peeked")  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    def __init__(self, iterable: Iterable[T]) -> None:
        self._it: Iterator[T] = iter(iterable)
        self._peeked: Option[T] = NONE

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> T:
        match self._peeked:
            case Some(value):
                self._peeked = NONE
                return value
            case Null():
                return next(self._it)

    def __bool__(self) -> bool:
        return self.peek().is_some()

    def peek(self) -> Option[T]:
        """Returns the `next()` value without advancing the `Iterator`.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if available, or `NONE` if the iteration is over.
        """
        match self._peeked:
            case Some(_):
                return self._peeked
            case Null():
                try:
                    self._peeked = option(next(self._it))
                except StopIteration:
                    return NONE
                else:
                    return self._peeked

    def next_if(self, func: Callable[[T], bool]) -> Option[T]:
        """Consume and return the next value of this iterator if a condition is `True`.

        Args:
            func (Callable[[T], bool]): A function that takes the next value and returns a boolean.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if the condition is true, or `NONE` if the condition is false or the iteration is over.

        Examples:
            Consume a number if it's equal to 0.
            ```python
            >>> from pyochain import Range
            >>> iterator = Range(0, 5).iter().peekable()
            >>> # The first item of the iterator is 0; consume it.
            >>> iterator.next_if(lambda x: x == 0)
            Some(0)
            >>> # The next item returned is now 1, so `next_if` will return `None`.
            >>> iterator.next_if(lambda x: x == 0)
            NONE
            >>> # `next_if` retains the next item if the predicate evaluates to `false` for it.
            >>> iterator.next()
            Some(1)

            ```
            Consume any number less than 10.
            ```python
            >>> iterator = Range(1, 20).iter().peekable()
            >>> # Consume all numbers less than 10
            >>> while iterator.next_if(lambda x: x < 10).is_some():
            ...     pass
            >>> # The next value returned will be 10
            >>> iterator.next()
            Some(10)
        """
        match self.next():
            case Some(matched) if func(matched):
                return Some(matched)
            case other:
                self._peeked = other
                return NONE

    def next_if_eq(self, expected: T) -> Option[T]:
        """Return the next item if it is equal to expected.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if it is equal to expected, or `NONE` if it is not equal or the iteration is over.

        Example:
            Consume a number if it's equal to 0.
            ```python
            >>> from pyochain import Range
            >>> iterator = Range(0, 5).iter().peekable()
            >>> # The first item of the iterator is 0; consume it.
            >>> iterator.next_if_eq(0)
            Some(0)
            >>> # The next item returned is now 1, so `next_if_eq` will return `None`.
            >>> iterator.next_if_eq(0)
            NONE
            >>> # `next_if_eq` retains the next item if it was not equal to `expected`.
            >>> iterator.next()
            Some(1)

            ```
        """
        return self.next_if(lambda nxt: nxt == expected)

    def next_if_map[R](
        self,
        f: Callable[[T], Result[R, T]],
    ) -> Option[R]:
        """Consumes the next value of this `Iterator` and applies a function *f* on it, returning the result if the closure returns `Ok`.

        Otherwise if the closure returns `Err` the value is put back for the next iteration.

        The content of the `Err` variant is typically the original value of the closure, but this is not required.

        If a different value is returned, the next `peek()` or `next()` call will result in this new value.

        Args:
            f (Callable[[T], Result[R, T]]): A function that takes the next value and returns a Result.

        Returns:
            Option[R]: The result of the function wrapped in `Some(R)` if the function returns `Ok(R)`, or `NONE` if the function returns `Err(T)` or the iteration is over.

        Examples:
            Parse the leading decimal number from an iterator of characters.
            ```python
            >>> from pyochain import Iter, Option, Some, NONE
            >>> import unicodedata
            >>>
            >>> iterator = Iter("125 GOTO 10").peekable()
            >>> line_num = 0
            >>> def try_parse_digit(c: str) -> Result[int, str]:
            ...     try:
            ...         res = Some(unicodedata.digit(c))
            ...     except ValueError as e:
            ...         res = NONE
            ...     return res.ok_or(c)
            >>>
            >>> digit = iterator.next_if_map(try_parse_digit)
            >>> while digit.is_some():
            ...     line_num = line_num * 10 + digit.unwrap()
            ...     digit = iterator.next_if_map(try_parse_digit)
            >>> line_num
            125
            >>> iterator.join("")
            ' GOTO 10'

            ```
        """
        match self.next():
            case Some(item):
                match f(item):
                    case Ok(result):
                        return Some(result)
                    case Err(item):
                        unpeek = Some(item)
            case Null():
                unpeek = NONE

        self._peeked = unpeek
        return NONE
