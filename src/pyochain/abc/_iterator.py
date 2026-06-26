from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, overload

from .._abc import (  # pyright: ignore[reportMissingModuleSource]
    PyoIteratorRS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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
