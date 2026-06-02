from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
)
from typing import TYPE_CHECKING, Any, Self, override

from . import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from ._seq import Seq
from .abc import PyoIterator

if TYPE_CHECKING:
    from .rs import Option


class Iter[T](PyoIterator[T]):
    """Concrete implementation for `abc::PyoIterator`.

    Can be instantiated from any `Iterable` (like lists, sets, generators, etc.) efficiently (it only calls the builtin `iter()` on the input).

    As such, creating an `Iter` from an `Iterator` is virtually free.

    Tip:
        `Iter::__iter__()` returns the underlying wrapped `Iterator`, hence native speed is kept.

        i.e `Iter([...]).map(f).collect(list)` is as fast as `list(map(f, [...]))`.

    Args:
        data (Iterable[T]): Any object that can be iterated over.

    See Also:
        [`abc::PyoIterator`][PyoIterator]: The abstract base class that `Iter` implements.

    Example:
        ```python
        >>> data = (0, 1, 2, 3, 4)
        >>> Iter(data).collect()
        Seq(0, 1, 2, 3, 4)
        >>> iterator = Iter(data)
        >>> # First we have a tuple iterator
        >>> iterator._inner.__class__.__name__
        'tuple_iterator'
        >>> # Now we have a map object
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped._inner.__class__.__name__
        'map'
        >>> # We collect it, by default into a Seq
        >>> mapped.collect()
        Seq(0, 2, 4, 6, 8)
        >>> # iterator is now exhausted
        >>> iterator.collect()
        Seq()

        ```
        You can also easily create an `Iter` from a generator expression:
        ```python
        >>> from pyochain import Iter
        >>> gen_expr = (x * x for x in range(5))
        >>> Iter(gen_expr).collect()
        Seq(0, 1, 4, 9, 16)

        ```
        Or from a generator function:
        ```python
        >>> from pyochain import Iter
        >>> def gen_func():
        ...     for x in range(5):
        ...         yield x * x
        >>>
        >>> Iter(gen_func()).collect()
        Seq(0, 1, 4, 9, 16)

        ```
    """

    _inner: Iterator[T]
    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = iter(data)

    @override
    def __iter__(self) -> Iterator[T]:
        return self._inner

    @override
    def __next__(self) -> T:
        return next(self._inner)

    def __bool__(self) -> bool:
        """Check if the `Iterator` has at least one element (mutates **self**).

        After calling this, the `Iterator` still contains all elements.

        Returns:
            bool: True if the `Iterator` has at least one element, False otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> it = Iter((1, 2, 3))
            >>> bool(it)
            True
            >>> it.collect()  # All elements still available
            Seq(1, 2, 3)

            ```
        """
        first = tuple(itertools.islice(self._inner, 1))
        self._inner = itertools.chain(first, self._inner)
        return len(first) > 0

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inner.__repr__()})"

    @classmethod
    def from_ref(cls, other: Self) -> Self:
        """Create an independent lazy copy from another `Iter`.

        Both the original and the returned `Iter` can be consumed independently, in a lazy manner.

        Note:
            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

            However, once both iterators have passed a value, it's freed from memory.

        See Also:
            - [`Iter::cloned`][cloned] which is the instance method version of this function.

        Args:
            other (Self): An `Iter` instance to copy.

        Returns:
            Self: A new `Iter` instance that is independent from the original.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> original = Iter((1, 2, 3))
            >>> copy = Iter.from_ref(original)
            >>> copy.map(lambda x: x * 2).collect()
            Seq(2, 4, 6)
            >>> original.next()
            Some(1)

            ```
        """
        it1, it2 = itertools.tee(other._inner)
        other._inner = it1
        return cls(it2)

    @staticmethod
    def once[V](value: V) -> Iter[V]:
        """Create an `Iter` that yields a single value.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.

        This can be considered the equivalent of `.insert()` but as a constructor.

        Args:
            value (V): The single value to yield.

        Returns:
            Iter[V]: An iterator yielding the specified value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter.once(42).collect()
            Seq(42,)

            ```
        """
        return Iter((value,))

    @staticmethod
    def once_with[**P, R](
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Iter[R]:
        """Create an `Iter`  that lazily generates a value exactly once by invoking the provided closure.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.

        This can be considered the equivalent of [`PyoIterator::insert`][PyoIterator.insert] but as a constructor.

        Unlike `.once()`, this function will lazily generate the value on request.

        Args:
            func (Callable[P, R]): The single value to yield.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Iter[R]: An iterator yielding the specified value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter.once_with(lambda: 42).collect()
            Seq(42,)

            ```
        """

        def _once_with() -> Generator[R]:
            yield func(*args, **kwargs)

        return Iter(_once_with())

    @staticmethod
    def from_count(start: int = 0, step: int = 1) -> Iter[int]:
        """Create an infinite `Iterator` of evenly spaced values.

        Warning:
            This creates an infinite iterator.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        Args:
            start (int): Starting value of the sequence.
            step (int): Difference between consecutive values.

        Returns:
            Iter[int]: An iterator generating the sequence.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter.from_count(10, 2).take(3).collect()
            Seq(10, 12, 14)

            ```
        """
        return Iter(itertools.count(start, step))

    @staticmethod
    def from_fn[**P, R](
        f: Callable[P, Option[R]], *args: P.args, **kwargs: P.kwargs
    ) -> Iter[R]:
        """Create an `Iterator` from a generator function.

        The `Callable` must return:

        - `Some(value)` to yield a value
        - `NONE` to stop the iteration

        You could consider this as a way to create an `Iterator` where the `__next__()` is the `__call__()` method.

        As such, you can either provide lambdas, partials, closures, or pre-existing classes where `__call__()` is implemented, but a `__next__()` is not desired.

        If you do have an `Iterator` class, simply pass it to the regular constructor, as this will be more efficient, ergonomic and idiomatic.

        Args:
            f (Callable[P, Option[R]]): `Callable` that returns the next item wrapped in `Option`.
            *args (P.args): Positional arguments to pass to **f**.
            **kwargs (P.kwargs): Keyword arguments to pass to **f**.

        Returns:
            Iter[R]: An iterator yielding values produced by **f**.

        Note:
            In Rust, this avoids defining a full struct and implementing `Iterator` for it when you have simple logic to generate values.

            This is implemented for "Rust API compliance", but in Python, generators comprehensions/functions with `yield` statements are the ergonomic equivalent.

        Example:
            Closure with captured local variable:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> def make_counter(max_val: int):
            ...     counter = 0
            ...
            ...     def gen() -> Option[int]:
            ...         nonlocal counter
            ...         counter += 1
            ...         return Some(counter) if counter <= max_val else NONE
            ...
            ...     return gen
            >>>
            >>> Iter.from_fn(make_counter(5)).collect()
            Seq(1, 2, 3, 4, 5)

            ```
            Stateful callable class:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Counter:
            ...     max: int
            ...     count: int = 0
            ...
            ...     def __call__(self) -> Option[int]:
            ...         self.count += 1
            ...         return Some(self.count) if self.count <= self.max else NONE
            >>>
            >>> Iter.from_fn(Counter(5)).collect()
            Seq(1, 2, 3, 4, 5)

            ```
            Simulated file/queue reader:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> from pyochain.collections import Deque
            >>>
            >>> def queue_consumer(items: Deque[int]) -> Callable[[], Option[int]]:
            ...     def consume() -> Option[int]:
            ...         return Some(items.pop_left()) if items else NONE
            ...
            ...     return consume
            >>>
            >>> Iter.from_fn(Deque([1, 2, 3]).into(queue_consumer)).collect()
            Seq(1, 2, 3)

            ```
        """
        return Iter(tls.FromFn(f, *args, **kwargs))

    @staticmethod
    def successors[U](first: Option[U], succ: Callable[[U], Option[U]]) -> Iter[U]:
        """Create an iterator of successive values computed from the previous one.

        The iterator yields `first` (if it is `Some`), then repeatedly applies **succ** to the
        previous yielded value until it returns `NONE`.

        Args:
            first (Option[U]): Initial item.
            succ (Callable[[U], Option[U]]): Successor function.

        Returns:
            Iter[U]: Iterator yielding `first` and its successors.

        Example:
            ```python
            >>> from pyochain import Iter, Some, NONE, Option
            >>>
            >>> def next_pow10(x: int) -> Option[int]:
            ...     return Some(x * 10) if x < 10_000 else NONE
            >>>
            >>> Iter.successors(Some(1), next_pow10).collect()
            Seq(1, 10, 100, 1000, 10000)
            >>> Iter.successors(NONE, next_pow10).collect()
            Seq()

            ```
        """
        return Iter(tls.Successors(first, succ))

    @staticmethod
    def from_repeat[O](obj: O, n: int | None = None) -> Iter[O]:
        """Repeat the provided object **n** times (as elements) in an `Iter`.

        If **n** is `None`, this will create an infinite `Iterator`.

        Be sure to use [`Iter::take`][Iter.take] or [`Iter::slice`][Iter.slice] to limit the number of items taken.

        Warning:
            Each repetition is a reference to the same object, not a copy.

            This means that if the object is mutable and you modify one of the repetitions, all next repetitions will reflect that change.

        Args:
            obj (O): The object to repeat.
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[O]: An `Iterator` of repeated **obj**.

        See Also:
            [`Iter::cycle`][cycle] to repeat the *elements* of the `Iterator`.
            [`Iter::repeat`][repeat] to repeat the *entire `Iterator`.

        Example:
            ```python
            >>> from pyochain import Seq, Iter
            >>> Iter.from_repeat(1, 3).collect()
            Seq(1, 1, 1)
            >>> Iter.from_repeat(("a", "b"), 2).collect()
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
        if n is None:
            return Iter(itertools.repeat(obj))
        return Iter(itertools.repeat(obj, n))

    @override
    def collect[R: Collection[Any]](
        self, collector: Callable[[Iterator[T]], R] = Seq[T]
    ) -> R:
        return collector(self._inner)

    def cloned(self) -> PyoIterator[T]:
        """Clone the `Iter` into a new independent `Iter` using `itertools.tee`.

        After calling this method, the original `Iter` will continue to yield elements independently of the cloned one.

        Note:
            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

            However, once both iterators have passed a value, it's freed from memory.

        Returns:
            PyoIterator[T]: A new independent cloned iterator.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> it = Iter((1, 2, 3))
            >>> cloned = it.cloned()
            >>> cloned.collect()
            Seq(1, 2, 3)
            >>> it.collect()
            Seq(1, 2, 3)

            ```
        """
        it1, it2 = itertools.tee(self._inner)
        self._inner = it1
        return self._from_iterable(it2)
