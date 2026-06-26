from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .._abc import (  # pyright: ignore[reportMissingModuleSource]
    PyoIteratorRS,
)

if TYPE_CHECKING:
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
        # NOTE: If we could know that the `Iterator` is double-ended, we could iterate from the end. In rust? easy. In python we don't have builtins to get the class, because Iterators are opaque most of the time. There's not abc with both `Iterator` and `Reversible`.
        return Deque.from_ref(deque(iter(self), n))
