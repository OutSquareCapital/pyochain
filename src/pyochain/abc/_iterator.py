from __future__ import annotations

import itertools
from abc import ABC
from typing import TYPE_CHECKING, Any, Self, overload, override

from .._abc import (  # pyright: ignore[reportMissingModuleSource]
    PyoIteratorRS,
)
from ..rs import NONE, Err, Null, Ok, Option, Result, Some, option

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from .._types import SupportsAnyRichComparison, SupportsComparison
    from .._vec import Vec
    from ..collections import Deque

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
    def map_with[T1, R](
        self, func: Callable[[T, T1], R], iterable: Iterable[T1], /
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, R](
        self,
        func: Callable[[T, T1, T2], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, R](
        self,
        func: Callable[[T, T1, T2, T3], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, R](
        self,
        func: Callable[[T, T1, T2, T3, T4], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, T5, R](
        self,
        func: Callable[[T, T1, T2, T3, T4, T5], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, T5, T6, R](
        self,
        func: Callable[[T, T1, T2, T3, T4, T5, T6], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        /,
    ) -> PyoIterator[R]: ...
    def map_with[R](
        self, func: Callable[..., R], *iterables: AnyIter
    ) -> PyoIterator[R]:
        """Applies a function to the elements of this `Iterator` and additional iterables.

        The provided function must take as many arguments as the number of iterables provided (including **self**).

        It is then applied to the items from all iterables in parallel.

        The `Iterator` stops when the shortest iterable is exhausted.

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
            >>> output = x.iter().map_with(Triangle, y, z).collect(Seq)
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
