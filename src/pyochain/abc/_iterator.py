from __future__ import annotations

import itertools
from abc import ABC
from typing import TYPE_CHECKING, Any, overload

from .._abc import (  # pyright: ignore[reportMissingModuleSource]
    PyoIteratorRS,
)
from ..rs import Option, option

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from .._peekable import Peekable
    from ..collections import Deque


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

            ```
        """
        from .._peekable import Peekable

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
        self, func: Callable[..., R], *iterables: Iterable[Any]
    ) -> PyoIterator[R]:
        """Applies a function to the elements of this `Iterator` and additional iterables.

        The provided function must take as many arguments as the number of iterables provided (including **self**).

        It is then applied to the items from all iterables in parallel.

        The `Iterator` stops when the shortest iterable is exhausted.

        Args:
            func (Callable[..., R]): Function to apply to the elements of the iterables.
            *iterables (Iterable[Any]): Additional iterables to zip with **self**.

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
            >>> x.iter().map_with(lambda a, b, c: a + b + c, y, z).collect(Seq)
            Seq(12, 15, 18)

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
