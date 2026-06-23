from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from ._utils import no_doctest
from .abc import PyoIterator
from .rs import Checkable, Fluent

@runtime_checkable
class PyoIterable[T](Fluent, Checkable, Iterable[T], Protocol):
    """Base ABC for all pyochain `Iterables`.

    It's the common API surface shared by:

    - eager `Collections`: `Seq`, `Vec`, `Set`, `SetMut`, `Dict`
    - lazy `Iterator`: `Iter`

    It extends the standard `Iterable[T]` protocol, as well as `Fluent` and `Checkable`.

    All concrete subclasses must implement `__iter__()`.

    Note:
        The difference between an `Iterable` and an `Iterator` is often misunderstood, but it's actually quite simple.

        An `Iterable` is any object that can **create** an `Iterator`.

        It's sole responsbility is to provide an `__iter__` method.

        This method must return an object that have a `__next__` method, which is the actual `Iterator`.

        An `Iterator` is an object that can produce elements one at a time, and can be exhausted.

        When you do a `for x in my_iterable`, Python implicitly calls `my_iterable.__iter__(), and then repeatedly calls `next()` on the resulting `Iterator` to get the elements.

        More concretely, a `list`, for example, is an `Iterable`.

        You can't call `next()` on a `list`, because it don't know how to produce elements by itself, it's primary responsibility being to **store** them.

        However, as soon as you call `map(my_list)`, `[x for x in my_list]`, (*my_list), or any other operation that needs to visit elements, an `Iterator` is created (implicitly or explicitly) from the `list`.

        It's also why `abc::Iterator::__iter__` returns `Self` by convention.

    Example:
        Since it's very straightforward to implement, it can very easily be integrated into business logic classes to provide them with a rich set of methods for free.

        ```python
        >>> from pyochain.abc import PyoIterable
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(slots=True)
        ... class ClientRegistry(PyoIterable[str]):
        ...     clients: list[str]
        ...
        ...     def __iter__(self):
        ...         return iter(self.clients)
        >>>
        >>> registry = ClientRegistry(["Alice", "Bob", "Charlie"])
        >>> registry.iter().all(lambda name: name.startswith("A"))
        False
        >>> registry.iter().join(", ")
        'Alice, Bob, Charlie'
        >>> registry.iter().map(str.lower).join(", ")
        'alice, bob, charlie'
        >>> registry.ok_or("Registry is empty").map(lambda s: s.iter().join(", "))
        Ok('Alice, Bob, Charlie')

        ```
    """

    def iter[I](self: PyoIterable[I]) -> PyoIterator[I]:
        """Returns a `PyoIterator` object over the `Iterable`.

        By default, this returns an `Iter`, but can be overriden by concrete subclasses.

        This method is the pyochain equivalent of the `__iter__` dunder method.

        Returns:
            PyoIterator[T]: An `Iterator` over the `Iterable`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> seq = Seq((1, 2, 3))
            >>> iterator = seq.iter()
            >>> iterator.collect(Seq)
            Seq(1, 2, 3)
            >>> # iterator is now empty
            >>> iterator.collect(Seq)
            Seq()

            ```
        """

class PyoIteratorRS[T](PyoIterable[T], Iterator[T], Protocol):
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

    @no_doctest
    @classmethod
    def _from_iterable[I](cls, iterable: Iterable[I]) -> PyoIterator[I]:
        """Internal constructor.

        Since some methods returns a new `PyoIterator`, we use this, with the assumption that the concrete subclass has an `__init__` that can accept an `Iterable[T]`.

        If you want to implement a different constructor, you will need to override this method with one that can construct new instances from an iterable argument.

        Args:
            iterable (Iterable[I]): An `Iterable` to create the new `PyoIterator` from.

        Returns:
            PyoIterator[I]: A new instance of the concrete `PyoIterator` subclass.

        See Also:
            This is how python standard library handle `collections::abc::Set`, see the first point below `Notes on using Set [...]`:

            https://docs.python.org/3/library/collections.abc.html#examples-and-recipes

        """
