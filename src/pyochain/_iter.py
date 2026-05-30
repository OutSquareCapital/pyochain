from __future__ import annotations

import functools
import itertools
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    ValuesView,
)
from enum import StrEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeGuard,
    TypeIs,
    overload,
    override,
)

from . import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from ._seq import Seq
from .abc import PyoIterator
from .rs import Option, option

if TYPE_CHECKING:
    from ._dict import Dict
    from ._range import Range
    from ._set import Set, SetMut
    from ._vec import Vec

type AnyOpt = Option[Any]  # pyright: ignore[reportExplicitAny]
type AnyIter = Iterable[Any]  # pyright: ignore[reportExplicitAny]
type ZippedLongest[T] = (
    Iter[tuple[Option[T], AnyOpt]]
    | Iter[tuple[Option[T], AnyOpt, AnyOpt]]
    | Iter[tuple[Option[T], AnyOpt, AnyOpt, AnyOpt]]
    | Iter[tuple[Option[T], AnyOpt, AnyOpt, AnyOpt, AnyOpt]]
    | Iter[tuple[AnyOpt, ...]]
)
"""Type representing the result of a `zip_longest` operation, which can yield tuples of varying lengths depending on the number of iterables zipped together."""
type FilterFn[T, R] = Callable[[T], bool | TypeIs[R] | TypeGuard[R]] | None
"""Optional closure that can be passed to `Iter::filter` to determine if an element should be yielded."""
# TODO: move to Rust the following:
# with_position, zip_longest, filter_star, repeat,  array_chunks # noqa: ERA001


class Position(StrEnum):
    """Type representing the position of an item in an iterable."""

    FIRST = auto()
    MIDDLE = auto()
    LAST = auto()
    ONLY = auto()


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

    def array_chunks(self, size: int) -> Iter[Self]:
        """Yield subiterators (chunks) that each yield a fixed number elements, determined by size.

        The last chunk will be shorter if there are not enough elements.

        Args:
            size (int): Number of elements in each chunk.

        Returns:
            Iter[Self]: An iterable of iterators, each yielding n elements.

        If the sub-iterables are read in order, the elements of *iterable*
        won't be stored in memory.

        If they are read out of order, :func:`itertools.tee` is used to cache
        elements as necessary.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> all_chunks = Iter.from_count().array_chunks(4)
            >>> c_1, c_2, c_3 = all_chunks.next(), all_chunks.next(), all_chunks.next()
            >>> c_2.unwrap().collect()  # c_1's elements have been cached; c_3's haven't been
            Seq(4, 5, 6, 7)
            >>> c_1.unwrap().collect()
            Seq(0, 1, 2, 3)
            >>> c_3.unwrap().collect()
            Seq(8, 9, 10, 11)

            ```
            You can collect the chunks into a collection of collections, for example:
            ```python
            >>> from pyochain import Seq
            >>> from pyochain.abc import PyoIterable
            >>> def collect_all_chunks(data: PyoIterable[int]) -> Seq[Seq[int]]:
            ...     return (
            ...         data.iter().array_chunks(3).map(lambda c: c.collect()).collect()
            ...     )
            >>> Seq((1, 2, 3, 4, 5, 6)).into(collect_all_chunks)
            Seq(Seq(1, 2, 3), Seq(4, 5, 6))
            >>> Seq((1, 2, 3, 4, 5, 6, 7, 8)).into(collect_all_chunks)
            Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8))

            ```
        """
        from collections import deque
        from contextlib import suppress

        def _chunks() -> Iterator[Self]:
            def _ichunk(
                iterator: Iterator[T], n: int
            ) -> tuple[Iterator[T], Callable[[int], int]]:
                cache: deque[T] = deque()
                chunk = itertools.islice(iterator, n)

                def _generator() -> Iterator[T]:
                    with suppress(StopIteration):
                        while True:
                            if cache:
                                yield cache.popleft()
                            else:
                                yield next(chunk)

                def _materialize_next(n: int) -> int:
                    to_cache = n - len(cache)

                    # materialize up to n
                    if to_cache > 0:
                        cache.extend(itertools.islice(chunk, to_cache))

                    # return number materialized up to n
                    return min(n, len(cache))

                return (_generator(), _materialize_next)

            new = self.__class__
            while True:
                # Create new chunk
                chunk, materialize_next = _ichunk(self._inner, size)

                # Check to see whether we're at the end of the source iterable
                if not materialize_next(size):
                    return

                yield new(chunk)
                _ = materialize_next(size)

        return Iter(_chunks())

    @overload
    def flatten[U](self: Iter[KeysView[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Iterable[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Generator[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[ValuesView[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Iterator[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Collection[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Sequence[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[list[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[tuple[U, ...]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Iter[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Seq[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Set[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[SetMut[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Vec[U]]) -> Iter[U]: ...
    @overload
    def flatten(self: Iter[range]) -> Iter[int]: ...
    @overload
    def flatten(self: Iter[Range]) -> Iter[int]: ...
    @overload
    def flatten[U](self: Iter[Dict[U, Any]]) -> Iter[U]: ...  # pyright: ignore[reportExplicitAny]
    def flatten[U: AnyIter](self: Iter[U]) -> Iter[Any]:  # pyright: ignore[reportExplicitAny]
        """Creates an `Iter` that flattens nested structures.

        This is useful when you have an `Iter` of `Iterable` and you want to remove one level of indirection.

        Returns:
            Iter[Any]: An `Iter` of flattened elements.


        Example:
            Basic usage:
            ```python
            >>> from pyochain import Iter
            >>> data = ((1, 2, 3, 4), (5, 6))
            >>> flattened = Iter(data).flatten().collect()
            >>> flattened
            Seq(1, 2, 3, 4, 5, 6)

            ```
            Mapping and then flattening:
            ```python
            >>> from pyochain import Iter
            >>> words = Iter(("alpha", "beta", "gamma"))
            >>> merged = words.flatten().collect()
            >>> merged
            Seq('a', 'l', 'p', 'h', 'a', 'b', 'e', 't', 'a', 'g', 'a', 'm', 'm', 'a')

            ```
            Flattening only removes one level of nesting at a time:
            ```python
            >>> from pyochain import Iter
            >>> d3 = (((1, 2), (3, 4)), ((5, 6), (7, 8)))
            >>> d2 = Iter(d3).flatten().collect()
            >>> d2
            Seq((1, 2), (3, 4), (5, 6), (7, 8))
            >>> d1 = Iter(d3).flatten().flatten().collect()
            >>> d1
            Seq(1, 2, 3, 4, 5, 6, 7, 8)

            ```
            Here we see that `flatten()` does not perform a “deep” flatten.

            Instead, only **one** level of nesting is removed.

            That is, if you `flatten()` a three-dimensional array, the result will be two-dimensional and not one-dimensional.

            To get a one-dimensional structure, you have to `flatten()` again.

        """
        return Iter(itertools.chain.from_iterable(self._inner))

    def flat_map[R](self, func: Callable[[T], Iterable[R]]) -> Iter[R]:
        """Creates an iterator that applies a function to each element of the original iterator and flattens the result.

        This is useful when the **func** you want to pass to `.map()` itself returns an iterable, and you want to avoid having nested iterables in the output.

        This is equivalent to calling `.map(func).flatten()`.

        Args:
            func (Callable[[T], Iterable[R]]): Function to apply to each element.

        Returns:
            Iter[R]: An iterable of flattened transformed elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).flat_map(lambda x: range(x)).collect()
            Seq(0, 0, 1, 0, 1, 2)

            ```
        """
        return Iter(itertools.chain.from_iterable(map(func, self._inner)))

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

    def map[R](self, func: Callable[[T], R]) -> Iter[R]:
        """Apply a function **func** to each element of the `Iter`.

        If you are good at thinking in types, you can think of `Iter.map()` like this:

        - You have an `Iterator` that gives you elements of some type `A`
        - You want an `Iterator` of some other type `B`
        - Thenyou can use `.map()`, passing a closure **func** that takes an `A` and returns a `B`.

        `Iter.map()` is conceptually similar to a for loop.

        However, as `Iter.map()` is lazy, it is best used when you are already working with other `Iter` instances.

        If you are doing some sort of looping for a side effect, it is considered more idiomatic to use `Iter.for_each()` than `Iter.map().collect()`.

        Args:
            func (Callable[[T], R]): Function to apply to each element.

        Returns:
            Iter[R]: An iterator of transformed elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).map(lambda x: x + 1).collect()
            Seq(2, 3)
            >>> # You can use methods on the class rather than on instance for convenience:
            >>> data = Seq(("a", "b", "c"))
            >>> data.iter().map(str.upper).collect()
            Seq('A', 'B', 'C')
            >>> data.iter().map(lambda s: s.upper()).collect()
            Seq('A', 'B', 'C')

            ```
        """
        return Iter(map(func, self._inner))

    @overload
    def map_star[R](
        self: Iter[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], R],  # pyright: ignore[reportExplicitAny]
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, R](
        self: Iter[tuple[T1, T2]],
        func: Callable[[T1, T2], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, R](
        self: Iter[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, R](
        self: Iter[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, R](
        self: Iter[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
    ) -> Iter[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
    ) -> Iter[R]: ...
    def map_star[U: AnyIter, R](self: Iter[U], func: Callable[..., R]) -> Iter[R]:
        """Applies a function to each element.where each element is an iterable.

        Unlike `.map()`, which passes each element as a single argument, `.starmap()` unpacks each element into positional arguments for the function.

        In short, for each element in the `Iter`, it computes `func(*element)`.

        Note:
            Always prefer using `.map_star()` over `.map()` when working with `Iter` of `tuple` elements.
            Not only it is more readable, but it's also much more performant (up to 30% faster in benchmarks).

        Args:
            func (Callable[..., R]): Function to apply to unpacked elements.

        Returns:
            Iter[R]: An iterable of results from applying the function to unpacked elements.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> def make_sku(color: str, size: str) -> str:
            ...     return f"{color}-{size}"
            >>> data = Seq(("blue", "red"))
            >>> data.iter().product(["S", "M"]).map_star(make_sku).collect()
            Seq('blue-S', 'blue-M', 'red-S', 'red-M')
            >>> # This is equivalent to:
            >>> data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).collect()
            Seq('blue-S', 'blue-M', 'red-S', 'red-M')

            ```
        """
        return Iter(itertools.starmap(func, self._inner))

    @overload
    def map_with[T1, R](
        self, iterable: Iterable[T1], /, *, func: Callable[[T, T1], R]
    ) -> Iter[R]: ...
    @overload
    def map_with[T1, T2, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        func: Callable[[T, T1, T2], R],
    ) -> Iter[R]: ...
    @overload
    def map_with[T1, T2, T3, R](
        self,
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        func: Callable[[T, T1, T2, T3], R],
    ) -> Iter[R]: ...
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
    ) -> Iter[R]: ...
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
    ) -> Iter[R]: ...
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
    ) -> Iter[R]: ...
    def map_with[R](self, *iterables: AnyIter, func: Callable[..., R]) -> Iter[R]:
        """Applies a function to the elements of this `Iterator` and additional iterables.

        The provided function must take as many arguments as the number of iterables provided (including **self**).

        It is then applied to the items from all iterables in parallel.

        the iterator stops when the shortest iterable is exhausted.

        Args:
            *iterables (AnyIter): Additional iterables to zip with **self**.
            func (Callable[..., R]): Function to apply to the elements of the iterables.

        Returns:
            Iter[R]: An `Iterator` of results from applying the function to the elements of the iterables.

        See Also:
            [`Iter::map_juxt`][map_juxt] to apply multiple functions to the same elements of the `Iterator`.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Triangle:
            ...     x: int
            ...     y: int
            ...     z: int
            >>>
            >>> x_list = [1, 2, 3]
            >>> y_list = [4, 5, 6]
            >>> z_list = [7, 8, 9]
            >>> output = Iter(x_list).map_with(y_list, z_list, func=Triangle).collect()
            >>> output
            Seq(Triangle(x=1, y=4, z=7), Triangle(x=2, y=5, z=8), Triangle(x=3, y=6, z=9))

            ```
        """
        return Iter(map(func, self._inner, *iterables))

    def map_while[R](self, func: Callable[[T], Option[R]]) -> Iter[R]:
        """Creates an `Iterator` that both yields elements based on a predicate and maps.

        `map_while()` takes a closure as an argument.

        It will call this closure on each element of the `Iterator`, and yield elements while it returns `Some(_)`.

        After `NONE` is returned, `Iter::map_while` stops and the rest of the elements are ignored.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element that returns `Option[R]`.

        Returns:
            Iter[R]: An `Iterator` of transformed elements until `NONE` is encountered.

        Example:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> def checked_div(x: int) -> Option[int]:
            ...     return Some(16 // x) if x != 0 else NONE
            >>>
            >>> data = Iter((-1, 4, 0, 1))
            >>> data.map_while(checked_div).collect()
            Seq(-16, 4)
            >>> data = Iter((0, 1, 2, -3, 4, 5, -6))
            >>> # Convert to positive ints, stop at first negative
            >>> data.map_while(lambda x: Some(x) if x >= 0 else NONE).collect()
            Seq(0, 1, 2)

            ```
        """
        return Iter(tls.MapWhile(self._inner, func))

    def repeat(self, n: int | None = None) -> Iter[Self]:
        """Repeat the entire `Iter` **n** times (as elements).

        If **n** is `None`, repeat indefinitely.

        Operates lazily, hence if you need to get the underlying elements, you will need to collect each repeated `Iter` via `.map(lambda x: x.collect())` or similar.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            [`Iter::cycle`][cycle] to repeat the *elements* of the `Iter` indefinitely.

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[Self]: An `Iter` of repeated `Iter`.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).repeat(3).map(list).collect()
            Seq([1, 2], [1, 2], [1, 2])

            ```
        """
        new = self.__class__

        def _repeat_infinite() -> Generator[Self]:
            tee = functools.partial(itertools.tee, self._inner, 1)
            iterators = tee()
            while True:
                yield new(iterators[0])
                iterators = tee()

        if n is None:
            return Iter(_repeat_infinite())
        return Iter(map(new, itertools.tee(self._inner, n)))

    def scan[U](self, initial: U, func: Callable[[U, T], Option[U]]) -> Iter[U]:
        """Transform elements by sharing state between iterations.

        `scan` takes two arguments:

            - an **initial** value which seeds the internal state
            - a **func** with two arguments

        The first being a reference to the internal state and the second an iterator element.

        The **func** can assign to the internal state to share state between iterations.

        On iteration, the **func** will be applied to each element of the iterator and the return value from the func, an Option, is returned by the next method.

        Thus the **func** can return `Some(value)` to yield value, or `NONE` to end the iteration.

        Args:
            initial (U): Initial state.
            func (Callable[[U, T], Option[U]]): Function that takes the current state and an item, and returns an Option.

        Returns:
            Iter[U]: An iterable of the yielded values.

        Example:
            ```python
            >>> from pyochain import Iter, Some, NONE
            >>> def accumulate_until_limit(state: int, item: int) -> Option[int]:
            ...     new_state = state + item
            ...     match new_state:
            ...         case _ if new_state <= 10:
            ...             return Some(new_state)
            ...         case _:
            ...             return NONE
            >>> Iter((1, 2, 3, 4, 5)).scan(0, accumulate_until_limit).collect()
            Seq(1, 3, 6, 10)

            ```
        """
        return Iter(tls.Scan(self._inner, initial, func))

    @overload
    def filter[N](self: Iter[N | None], func: None = None) -> Iter[N]: ...
    @overload
    def filter[R](self, func: Callable[[T], TypeIs[R]]) -> Iter[R]: ...
    @overload
    def filter[R](self, func: Callable[[T], TypeGuard[R]]) -> Iter[R]: ...
    @overload
    def filter(self, func: Callable[[T], bool] | None) -> Self: ...
    def filter[R, N](self, func: FilterFn[T, R] = None) -> Self | Iter[R] | Iter[N]:
        """Creates an `Iter` with an optional closure to determine if an element should be yielded.

        Given an element the closure must return `True` or `False`.

        The returned `Iter` will yield only the elements for which the closure returns `True`.

        If no closure is provided, the elements are directly evaluated on their truthiness.

        This means that empty collections, `0`, `False`, and `None` will be filtered out.

        The closure can return a `TypeIs` or `TypeGuard` to narrow the type of the returned `Iterator`.

        This won't have any runtime effect, but allows for better type inference.

        Note:
            `Iter.filter(f).next()` is equivalent to `Iter.find(f)`.

        Args:
            func (FilterFn[T, R]): Function to evaluate each item.

        Returns:
            Self | Iter[R] | Iter[N]: An `Iterator` of the items that satisfy the predicate.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = (1, 2, 3)
            >>> Iter(data).filter(lambda x: x > 1).collect()
            Seq(2, 3)
            >>> # See the equivalence of next and find:
            >>> Iter(data).filter(lambda x: x > 1).next()
            Some(2)
            >>> Iter(data).find(lambda x: x > 1)
            Some(2)
            >>> # Using TypeIs to narrow type:
            >>> from typing import TypeIs
            >>> def _is_str(x: object) -> TypeIs[str]:
            ...     return isinstance(x, str)
            >>> mixed_data = (1, "two", 3.0, "four")
            >>> Iter(mixed_data).filter(_is_str).collect()
            Seq('two', 'four')
            >>> maybe_none = (1, None, 3, None)
            >>> Iter(maybe_none).filter().collect()
            Seq(1, 3)
            >>> maybe_false = (0, 1, False, 2, "", 3, None)
            >>> Iter(maybe_false).filter().collect()
            Seq(1, 2, 3)

            ```
        """
        return self.__class__(filter(func, self._inner))

    @overload
    def filter_star(
        self: Iter[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], bool],  # pyright: ignore[reportExplicitAny]
    ) -> Iter[tuple[Any]]: ...  # pyright: ignore[reportExplicitAny]
    @overload
    def filter_star[T1, T2](
        self: Iter[tuple[T1, T2]],
        func: Callable[[T1, T2], bool],
    ) -> Iter[tuple[T1, T2]]: ...
    @overload
    def filter_star[T1, T2, T3](
        self: Iter[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], bool],
    ) -> Iter[tuple[T1, T2, T3]]: ...
    @overload
    def filter_star[T1, T2, T3, T4](
        self: Iter[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], bool],
    ) -> Iter[tuple[T1, T2, T3, T4]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5](
        self: Iter[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5, T6]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5, T6, T7]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8, T9](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], bool],
    ) -> Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]]: ...

    def filter_star[U: AnyIter](self: Iter[U], func: Callable[..., bool]) -> Iter[U]:
        """Creates an `Iter` which uses a closure **func** to determine if an element should be yielded, where each element is an iterable.

        Unlike `.filter()`, which passes each element as a single argument, `.filter_star()` unpacks each element into positional arguments for the **func**.

        In short, for each element in the `Iter`, it computes `func(*element)`.

        This is useful after using methods like `.zip()`, `.product()`, or `.enumerate()` that yield tuples.

        Args:
            func (Callable[..., bool]): Function to evaluate unpacked elements.

        Returns:
            Iter[U]: An `Iter` of the items that satisfy the predicate.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq(("apple", "banana", "cherry", "date"))
            >>> output = (
            ...     data.iter()
            ...     .enumerate()
            ...     .filter_star(lambda index, _: index % 2 == 0)
            ...     .map_star(lambda _, fruit: fruit.title())
            ...     .collect()
            ... )
            >>> output
            Seq('Apple', 'Cherry')

            ```
        """
        return Iter(filter(lambda x: func(*x), self._inner))

    @overload
    def filter_false[N](self: Iter[N | None], func: None = None) -> Iter[None]: ...
    @overload
    def filter_false[U](self, func: Callable[[T], TypeIs[U]]) -> Iter[U]: ...
    @overload
    def filter_false[U](self, func: Callable[[T], TypeGuard[U]]) -> Iter[U]: ...
    @overload
    def filter_false(self, func: Callable[[T], bool]) -> Iter[T]: ...
    def filter_false[U](self, func: FilterFn[T, U] = None) -> Iter[T] | Iter[U]:
        """Return elements for which **func** is `False`.

        The **func** can return a `TypeIs` to narrow the type of the returned `Iter`.

        This won't have any runtime effect, but allows for better type inference.

        Args:
            func (FilterFn[T, U]): Function to evaluate each item.

        Returns:
            Iter[T] | Iter[U]: An `Iter` of the items that do not satisfy the predicate.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).filter_false(lambda x: x > 1).collect()
            Seq(1,)

            ```
        """
        return Iter(itertools.filterfalse(func, self._inner))

    def filter_map[R](self, func: Callable[[T], Option[R]]) -> Iter[R]:
        """Creates an iterator that both filters and maps.

        The returned iterator yields only the values for which the supplied closure returns Some(value).

        `filter_map` can be used to make chains of `filter` and map more concise.

        The example below shows how a `map().filter().map()` can be shortened to a single call to `filter_map`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each item.

        Returns:
            Iter[R]: An iterable of the results where func returned `Some`.

        See Also:
            [`Iter::filter`][filter] with no closure provided if you want to filter out Python native `None` values.

        Example:
            ```python
            >>> from pyochain import Result, Ok, Err, Seq
            >>> def _parse(s: str) -> Result[int, str]:
            ...     try:
            ...         return Ok(int(s))
            ...     except ValueError:
            ...         return Err(f"Invalid integer, got {s!r}")
            >>>
            >>> data = Seq(("1", "two", "NaN", "four", "5"))
            >>> parsed = data.iter().filter_map(lambda s: _parse(s).ok()).collect()
            >>> parsed
            Seq(1, 5)
            >>> # Equivalent to:
            >>> parsed = (
            ...     data.iter()
            ...     .map(lambda s: _parse(s).ok())
            ...     .filter(lambda s: s.is_some())
            ...     .map(lambda s: s.unwrap())
            ...     .collect()
            ... )
            >>> parsed
            Seq(1, 5)

            ```
        """
        return Iter(tls.FilterMap(self._inner, func))

    @overload
    def filter_map_star[R](
        self: Iter[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], Option[R]],  # pyright: ignore[reportExplicitAny]
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, R](
        self: Iter[tuple[T1, T2]],
        func: Callable[[T1, T2], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, R](
        self: Iter[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, R](
        self: Iter[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, R](
        self: Iter[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], Option[R]],
    ) -> Iter[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Iter[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], Option[R]],
    ) -> Iter[R]: ...
    def filter_map_star[U: AnyIter, R](
        self: Iter[U], func: Callable[..., Option[R]]
    ) -> Iter[R]:
        """Creates an iterator that both filters and maps, where each element is an iterable.

        Unlike `.filter_map()`, which passes each element as a single argument, `.filter_map_star()` unpacks each element into positional arguments for the function.

        In short, for each `element` in the sequence, it computes `func(*element)`.

        This is useful after using methods like `zip`, `product`, or `enumerate` that yield tuples.

        Args:
            func (Callable[..., Option[R]]): Function to apply to unpacked elements.

        Returns:
            Iter[R]: An iterable of the results where func returned `Some`.

        Example:
            ```python
            >>> from pyochain import Iter, Result, Ok, Err
            >>> data = (("1", "10"), ("two", "20"), ("3", "thirty"))
            >>> def _parse_pair(s1: str, s2: str) -> Result[tuple[int, int], str]:
            ...     try:
            ...         return Ok((int(s1), int(s2)))
            ...     except ValueError:
            ...         return Err(f"Invalid integer pair: {s1!r}, {s2!r}")
            >>>
            >>> parsed = (
            ...     Iter(data)
            ...     .filter_map_star(lambda s1, s2: _parse_pair(s1, s2).ok())
            ...     .collect()
            ... )
            >>> parsed
            Seq((1, 10),)

            ```
        """
        return Iter(tls.FilterMapStar(self._inner, func))

    @overload
    def zip[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        strict: bool = ...,
    ) -> Iter[tuple[T, T1]]: ...
    @overload
    def zip[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        strict: bool = ...,
    ) -> Iter[tuple[T, T1, T2]]: ...
    @overload
    def zip[T1, T2, T3](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        strict: bool = ...,
    ) -> Iter[tuple[T, T1, T2, T3]]: ...
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
    ) -> Iter[tuple[T, T1, T2, T3, T4]]: ...
    def zip(self, *others: AnyIter, strict: bool = False) -> Iter[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
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
            Iter[tuple[Any, ...]]: An `Iter` of tuples containing elements from the zipped Iter and other iterables.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).zip((10, 20)).collect()
            Seq((1, 10), (2, 20))
            >>> Iter(("a", "b")).zip((1, 2, 3)).collect()
            Seq(('a', 1), ('b', 2))

            ```
        """
        return Iter(zip(self._inner, *others, strict=strict))

    @overload
    def zip_longest[T2](
        self, iter2: Iterable[T2], /
    ) -> Iter[tuple[Option[T], Option[T2]]]: ...
    @overload
    def zip_longest[T2, T3](
        self, iter2: Iterable[T2], iter3: Iterable[T3], /
    ) -> Iter[tuple[Option[T], Option[T2], Option[T3]]]: ...
    @overload
    def zip_longest[T2, T3, T4](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
    ) -> Iter[tuple[Option[T], Option[T2], Option[T3], Option[T4]]]: ...
    @overload
    def zip_longest[T2, T3, T4, T5](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> Iter[
        tuple[
            Option[T],
            Option[T2],
            Option[T3],
            Option[T4],
            Option[T5],
        ]
    ]: ...
    @overload
    def zip_longest(
        self,
        iter2: Iterable[T],
        iter3: Iterable[T],
        iter4: Iterable[T],
        iter5: Iterable[T],
        iter6: Iterable[T],
        /,
        *iterables: AnyIter,
    ) -> Iter[tuple[Option[T], ...]]: ...
    def zip_longest(self, *others: AnyIter) -> ZippedLongest[T]:
        """Return a zip Iterator who yield a tuple where the i-th element comes from the i-th iterable argument.

        Yield values until the longest iterable in the argument sequence is exhausted, and then it raises StopIteration.

        The longest iterable determines the length of the returned iterator, and will return `Some[T]` until exhaustion.

        When the shorter iterables are exhausted, they yield `NONE`.

        Args:
            *others (AnyIter): Other iterables to zip with.

        Returns:
            ZippedLongest[T]: An iterable of tuples containing optional elements from the zipped iterables.

        Example:
            ```python
            >>> from pyochain import Iter, Some, NONE, Vec
            >>> Iter((1, 2)).zip_longest([10]).collect()
            Seq((Some(1), Some(10)), (Some(2), NONE))
            >>> # Can be combined with try collect to filter out the NONE:
            >>> zipped = (
            ...     Iter((1, 2))
            ...     .zip_longest([10])
            ...     .map(lambda x: Iter(x).try_collect())
            ...     .collect()
            ... )
            >>> zipped
            Seq(Some(Vec(1, 10)), NONE)

            ```
        """
        return Iter(
            tuple(option(t) for t in tup)
            for tup in itertools.zip_longest(self._inner, *others, fillvalue=None)
        )

    def unzip[U, V](self: Iter[tuple[U, V]]) -> tuple[Iter[U], Iter[V]]:
        """Converts an iterator of pairs into a pair of iterators.

        This function is, in some sense, the opposite of `.zip()`.

        Both iterators share the same underlying source.

        Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

        Returns:
            tuple[Iter[U], Iter[V]]: A tuple containing two iterators, one for each element of the pairs.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = ((1, "a"), (2, "b"), (3, "c"))
            >>> left, right = Iter(data).unzip()
            >>> left.collect()
            Seq(1, 2, 3)
            >>> right.collect()
            Seq('a', 'b', 'c')

            ```
        """
        left, right = itertools.tee(self._inner, 2)
        return Iter(x[0] for x in left), Iter(x[1] for x in right)

    def cloned(self) -> Self:
        """Clone the `Iter` into a new independent `Iter` using `itertools.tee`.

        After calling this method, the original `Iter` will continue to yield elements independently of the cloned one.

        Note:
            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

            However, once both iterators have passed a value, it's freed from memory.

        Returns:
            Self: A new independent cloned iterator.

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
        return self.__class__(it2)

    @overload
    def product(self) -> Iter[tuple[T]]: ...
    @overload
    def product[T1](self, iter1: Iterable[T1], /) -> Iter[tuple[T, T1]]: ...
    @overload
    def product[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
    ) -> Iter[tuple[T, T1, T2]]: ...
    @overload
    def product[T1, T2, T3](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
    ) -> Iter[tuple[T, T1, T2, T3]]: ...
    @overload
    def product[T1, T2, T3, T4](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
    ) -> Iter[tuple[T, T1, T2, T3, T4]]: ...

    def product(self, *others: AnyIter) -> Iter[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """Computes the Cartesian product with another iterable.

        This is the declarative equivalent of nested for-loops.

        It pairs every element from the source iterable with every element from the
        other iterable.

        Args:
            *others (AnyIter): Other iterables to compute the Cartesian product with.

        Returns:
            Iter[tuple[Any, ...]]: An iterable of tuples containing elements from the Cartesian product.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(["blue", "red"]).product(["S", "M"]).collect()
            Seq(('blue', 'S'), ('blue', 'M'), ('red', 'S'), ('red', 'M'))
            >>> res = (
            ...     Iter(["blue", "red"])
            ...     .product(["S", "M"])
            ...     .map_star(lambda color, size: f"{color}-{size}")
            ...     .collect()
            ... )
            >>> res
            Seq('blue-S', 'blue-M', 'red-S', 'red-M')
            >>> res = (
            ...     Iter((1, 2, 3))
            ...     .product([10, 20])
            ...     .filter_star(lambda a, b: a * b >= 40)
            ...     .map_star(lambda a, b: a * b)
            ...     .collect()
            ... )
            >>> res
            Seq(40, 60)
            >>> res = (
            ...     Iter([1])
            ...     .product(["a", "b"], [True])
            ...     .filter_star(lambda _a, b, _c: b != "a")
            ...     .map_star(lambda a, b, c: f"{a}{b} is {c}")
            ...     .collect()
            ... )
            >>> res
            Seq('1b is True',)

            ```
        """
        return Iter(itertools.product(self._inner, *others))

    @overload
    def map_windows[R](
        self, length: Literal[1], func: Callable[[tuple[T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[2], func: Callable[[tuple[T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[3], func: Callable[[tuple[T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[4], func: Callable[[tuple[T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[5], func: Callable[[tuple[T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[6], func: Callable[[tuple[T, T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[7], func: Callable[[tuple[T, T, T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[8], func: Callable[[tuple[T, T, T, T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[9], func: Callable[[tuple[T, T, T, T, T, T, T, T, T]], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self,
        length: Literal[10],
        func: Callable[[tuple[T, T, T, T, T, T, T, T, T, T]], R],
    ) -> Iter[R]: ...
    @overload
    def map_windows[R](
        self, length: int, func: Callable[[tuple[T, ...]], R]
    ) -> Iter[R]: ...
    def map_windows[R](
        self,
        length: int,
        func: Callable[[tuple[Any, ...]], R],  # pyright: ignore[reportExplicitAny]
    ) -> Iter[R]:
        """Calls the given *func* for each contiguous window of size *length* over **self**.

        The windows during mapping overlaps.

        The provided function is called with the entire window as a single tuple argument.

        Args:
            length (int): The length of each window.
            func (Callable[[tuple[Any, ...]], R]): Function to apply to each window.

        Returns:
            Iter[R]: An iterator over the outputs of func.

        See Also:
            [`Iter::map_windows_star`][map_windows_star] for a version that unpacks the window into separate arguments.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> import statistics
            >>> Iter((1, 2, 3, 4)).map_windows(2, statistics.mean).collect()
            Seq(1.5, 2.5, 3.5)
            >>> joined = (
            ...     Iter("abcd")
            ...     .map_windows(3, lambda window: "".join(window).upper())
            ...     .collect()
            ... )
            >>> joined
            Seq('ABC', 'BCD')
            >>> sum_windows = Iter((10, 20, 30, 40, 50)).map_windows(4, sum).collect()
            >>> sum_windows
            Seq(100, 140)

            ```
        """
        return Iter(map(func, tls.SlidingWindow(self._inner, length)))

    @overload
    def map_windows_star[R](
        self, length: Literal[1], func: Callable[[T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[2], func: Callable[[T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[3], func: Callable[[T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[4], func: Callable[[T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[5], func: Callable[[T, T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[6], func: Callable[[T, T, T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[7], func: Callable[[T, T, T, T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[8], func: Callable[[T, T, T, T, T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[9], func: Callable[[T, T, T, T, T, T, T, T, T], R]
    ) -> Iter[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[10], func: Callable[[T, T, T, T, T, T, T, T, T, T], R]
    ) -> Iter[R]: ...
    def map_windows_star[R](self, length: int, func: Callable[..., R]) -> Iter[R]:
        """Calls the given *func* for each contiguous window of size *length* over **self**.

        The windows during mapping overlaps.

        The provided function is called with each element of the window as separate arguments.

        Args:
            length (int): The length of each window.
            func (Callable[..., R]): Function to apply to each window.

        Returns:
            Iter[R]: An iterator over the outputs of func.

        See Also:
            [`Iter::map_windows`][map_windows] for a version that passes the entire window as a single tuple argument.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter("abcd").map_windows_star(2, lambda x, y: f"{x}+{y}").collect()
            Seq('a+b', 'b+c', 'c+d')
            >>> Iter([1, 2, 3, 4]).map_windows_star(2, lambda x, y: x + y).collect()
            Seq(3, 5, 7)

            ```
        """
        return Iter(itertools.starmap(func, tls.SlidingWindow(self._inner, length)))

    def batch(self, n: int, *, strict: bool = False) -> Iter[tuple[T, ...]]:
        """Batch elements into tuples of length n and return a new Iter.

        - The last batch may be shorter than n.
        - The data is consumed lazily, just enough to fill a batch.
        - The result is yielded as soon as a batch is full or when the input iterable is exhausted.

        Args:
            n (int): Number of elements in each batch.
            strict (bool): If `True`, raises a ValueError if the last batch is not of length n.

        Returns:
            Iter[tuple[T, ...]]: An iterable of batched tuples.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter("ABCDEFG").batch(3).collect()
            Seq(('A', 'B', 'C'), ('D', 'E', 'F'), ('G',))

            ```
        """
        return Iter(itertools.batched(self._inner, n, strict=strict))

    def enumerate(self, start: int = 0) -> Iter[tuple[int, T]]:
        """Return a `Iter` of (index, value) pairs.

        Each value in the `Iter` is paired with its index, starting from 0.

        Tip:
            `Iter.map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            start (int): The starting index.

        Returns:
            Iter[tuple[int, T]]: An `Iter` of (index, value) pairs.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = ("apple", "banana", "cherry")
            >>> output = Iter(data).enumerate().collect()
            >>> output
            Seq((0, 'apple'), (1, 'banana'), (2, 'cherry'))
            >>> output = (
            ...     Iter(data)
            ...     .enumerate()
            ...     .map_star(lambda idx, val: (idx, val.upper()))
            ...     .collect()
            ... )
            >>> output
            Seq((0, 'APPLE'), (1, 'BANANA'), (2, 'CHERRY'))

            ```
        """
        return Iter(enumerate(self._inner, start))

    @overload
    def combinations(self, r: Literal[2]) -> Iter[tuple[T, T]]: ...
    @overload
    def combinations(self, r: Literal[3]) -> Iter[tuple[T, T, T]]: ...
    @overload
    def combinations(self, r: Literal[4]) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def combinations(self, r: Literal[5]) -> Iter[tuple[T, T, T, T, T]]: ...
    def combinations(self, r: int) -> Iter[tuple[T, ...]]:
        """Return all combinations of length r.

        Args:
            r (int): Length of each combination.

        Returns:
            Iter[tuple[T, ...]]: An iterable of combinations.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).combinations(2).collect()
            Seq((1, 2), (1, 3), (2, 3))

            ```
        """
        return Iter(itertools.combinations(self._inner, r))

    @overload
    def permutations(self, r: Literal[2]) -> Iter[tuple[T, T]]: ...
    @overload
    def permutations(self, r: Literal[3]) -> Iter[tuple[T, T, T]]: ...
    @overload
    def permutations(self, r: Literal[4]) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def permutations(self, r: Literal[5]) -> Iter[tuple[T, T, T, T, T]]: ...
    def permutations(self, r: int | None = None) -> Iter[tuple[T, ...]]:
        """Return all permutations of length r.

        Args:
            r (int | None): Length of each permutation. Defaults to the length of the iterable.

        Returns:
            Iter[tuple[T, ...]]: An iterable of permutations.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).permutations(2).collect()
            Seq((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))

            ```
        """
        return Iter(itertools.permutations(self._inner, r))

    @overload
    def combinations_with_replacement(self, r: Literal[2]) -> Iter[tuple[T, T]]: ...
    @overload
    def combinations_with_replacement(self, r: Literal[3]) -> Iter[tuple[T, T, T]]: ...
    @overload
    def combinations_with_replacement(
        self,
        r: Literal[4],
    ) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def combinations_with_replacement(
        self,
        r: Literal[5],
    ) -> Iter[tuple[T, T, T, T, T]]: ...
    def combinations_with_replacement(self, r: int) -> Iter[tuple[T, ...]]:
        """Return all combinations with replacement of length r.

        Args:
            r (int): Length of each combination.

        Returns:
            Iter[tuple[T, ...]]: An iterable of combinations with replacement.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).combinations_with_replacement(2).collect()
            Seq((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))

            ```
        """
        return Iter(itertools.combinations_with_replacement(self._inner, r))

    def pairwise(self) -> Iter[tuple[T, T]]:
        """Return an iterator over pairs of consecutive elements.

        Returns:
            Iter[tuple[T, T]]: An iterable of pairs of consecutive elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).pairwise().collect()
            Seq((1, 2), (2, 3))

            ```
        """
        return Iter(itertools.pairwise(self._inner))

    @overload
    def map_juxt[R1, R2](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        /,
    ) -> Iter[tuple[R1, R2]]: ...
    @overload
    def map_juxt[R1, R2, R3](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        /,
    ) -> Iter[tuple[R1, R2, R3]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5, R6]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5, R6, R7]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5, R6, R7, R8]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8, R9](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        func9: Callable[[T], R9],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5, R6, R7, R8, R9]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        func9: Callable[[T], R9],
        func10: Callable[[T], R10],
        /,
    ) -> Iter[tuple[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]]: ...
    @overload
    def map_juxt[R](self, *funcs: Callable[[T], R]) -> Iter[tuple[R, ...]]: ...
    def map_juxt(self, *funcs: Callable[[T], Any]) -> Iter[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        """Apply several functions to each item of the `Iterator`.

        Returns a new `Iter` where each item is a tuple of the results of applying each function to the original item.

        This can be very handy to compute multiple transformations or properties of the same item in a single pass, without needing to iterate multiple times.

        As such, this can be considered as an alternative to various patterns, such as `Iter::{for_each, fold}` with mutable collections, or `Iter::map` followed by `Iter::zip` to combine the results.

        Args:
            *funcs (Callable[[T], Any]): Functions to apply to each item.

        Returns:
            Iter[tuple[Any, ...]]: An iterable of tuples containing the results of each function.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> def is_even(n: int) -> bool:
            ...     return n % 2 == 0
            >>> def is_positive(n: int) -> bool:
            ...     return n > 0
            >>>
            >>> Iter([1, -2, 3]).map_juxt(is_even, is_positive).collect()
            Seq((False, True), (True, False), (False, True))

            ```
            If you need to pass additional args and kwargs to the functions, you can use `functools::partial` or create curried functions like this:
            ```python
            >>> def curried_add(a: int) -> Callable[[int], int]:
            ...     def fn(b: int) -> int:
            ...         return a + b
            ...
            ...     return fn
            >>>
            >>> Iter((1, 2, 3)).map_juxt(curried_add(10), curried_add(20)).collect()
            Seq((11, 21), (12, 22), (13, 23))

            ```
            You can then combine this with various other methods to perform complex transformations in a clean and efficient way, without needing to iterate multiple times or create intermediate collections.

            Example with `filter_star`:
            ```python
            >>> from pyochain import Range
            >>> res = (
            ...     Range(0, 5)
            ...     .iter()
            ...     .map_juxt(lambda x: x * 2, lambda x: x**2)
            ...     .filter_star(lambda double, square: double + square <= 5)
            ...     .collect()
            ... )
            >>> res
            Seq((0, 0), (2, 1))

            ```
        """
        return Iter(map(tls.Juxt(*funcs), self._inner))

    def with_position(self) -> Iter[tuple[Position, T]]:
        """Return an iterable over (`Position`, `T`) tuples.

        The `Position` indicates whether the item `T` is the first, middle, last, or only element in the iterable.

        Returns:
            Iter[tuple[Position, T]]: An iterable of (`Position`, item) tuples.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(("a", "b", "c")).with_position().map_star(lambda pos, val: (pos.name, val)).collect()
            Seq(('FIRST', 'a'), ('MIDDLE', 'b'), ('LAST', 'c'))
            >>> Iter(["a"]).with_position().map_star(lambda pos, val: (pos.name, val)).collect()
            Seq(('ONLY', 'a'),)

            ```
        """

        def _gen(data: Iterator[T]) -> Iterator[tuple[Position, T]]:
            try:
                first = next(data)
            except StopIteration:
                return

            try:
                second = next(data)
            except StopIteration:
                yield (Position.ONLY, first)
                return
            yield (Position.FIRST, first)

            current: T = second
            for nxt in self._inner:
                yield (Position.MIDDLE, current)
                current = nxt
            yield (Position.LAST, current)

        return Iter(_gen(self._inner))

    @overload
    def group_by(self, key: None = None) -> Iter[tuple[T, Self]]: ...
    @overload
    def group_by[K](self, key: Callable[[T], K]) -> Iter[tuple[K, Self]]: ...
    @overload
    def group_by[K](
        self, key: Callable[[T], K] | None = None
    ) -> Iter[tuple[K, Self] | tuple[T, Self]]: ...
    def group_by(
        self,
        key: Callable[[T], Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> Iter[tuple[Any | T, Self]]:  # pyright: ignore[reportExplicitAny]
        """Make an `Iter` that returns consecutive keys and groups from the iterable.

        The values yielded are `(K, Self)` tuples, where the first element is the group key and the second element is an `Iter` of type `T` over the group values.

        The `Iter` needs to already be sorted on the same key function.

        This is due to the fact that it generates a new `Group` every time the value of the **key** function changes.

        That behavior differs from SQL's `GROUP BY` which aggregates common elements regardless of their input order.

        Warning:
            You must materialize the second element of the tuple immediately when iterating over groups.

            Because `.group_by()` uses Python's `itertools.groupby` under the hood, each group's iterator shares internal state.

            When you advance to the next group, the previous group's iterator becomes invalid and will yield empty results.

        Args:
            key (Callable[[T], Any] | None): Function computing a key value for each element..
        If not specified or is None, **key** defaults to an identity function and returns the element unchanged.

        Returns:
            Iter[tuple[Any | T, Self]]: An `Iter` of `(key, value)` tuples.

        Example:
            `group_by` can let you compute complex operations very easily and efficiently.

            For example, if we want to group even and odd numbers, we can do it like this:
            ```python
            >>> from pyochain import Iter, Dict
            >>> from operator import itemgetter
            >>> # Example 1: Group even and odd numbers
            >>> (
            ...     Iter.from_count()  # create an infinite iterator of integers
            ...     .take(8)  # take the first 8
            ...     .map(lambda x: (x % 2 == 0, x))  # map to (is_even, value)
            ...     .sort_by(itemgetter(0))  # sort by is_even
            ...     .iter()  # Since sort collect to a Vec, we need to convert back to Iter
            ...     .group_by(itemgetter(0))  # group by is_even
            ...     # extract values from groups, discarding keys, and materializing them
            ...     .map_star(
            ...         lambda g, vals: (g, vals.map_star(lambda _, y: y).collect())
            ...     )
            ...     .collect(Dict)
            ... )
            Dict(False: Seq(1, 3, 5, 7), True: Seq(0, 2, 4, 6))

            ```
            If we have a dataset who's items have a common key and who's already sorted by that key, we can easily perform grouped operations on it, like this:
            ```python
            >>> from pyochain import Iter
            >>> data = (
            ...     {"name": "Alice", "gender": "F"},
            ...     {"name": "Bob", "gender": "M"},
            ...     {"name": "Charlie", "gender": "M"},
            ...     {"name": "Dan", "gender": "M"},
            ... )
            >>> # group by the gender key, and count the number of people in each group
            >>> output = (
            ...     Iter(data)
            ...     .group_by(lambda x: x["gender"])
            ...     .map_star(lambda g, vals: (g, vals.count()))
            ...     .collect()
            ... )
            >>> output
            Seq(('F', 1), ('M', 3))

            ```
            However, you must be careful to materialize the group values immediately when iterating over groups, see below how the values of the groups are empty::
            ```python
            >>> from pyochain import Iter
            >>> groups = (
            ...     Iter(("a1", "a2", "b1"))
            ...     .group_by(lambda x: x[0])
            ...     .collect()
            ...     .iter()
            ...     .map_star(lambda g, vals: (g, vals.collect()))
            ...     .collect()
            ... )
            >>> groups
            Seq(('a', Seq()), ('b', Seq()))

            ```
            As such, the correct pattern is the following:
            ```python
            >>> from pyochain import Iter
            >>> groups = (
            ...     Iter(("a1", "a2", "b1", "b2"))
            ...     .group_by(lambda x: x[0])
            ...     .map_star(lambda g, vals: (g, vals.collect()))  # ✅ Materialize NOW
            ...     .collect()
            ...     .iter()
            ...     .collect()
            ... )
            >>> groups
            Seq(('a', Seq('a1', 'a2')), ('b', Seq('b1', 'b2')))

            ```
        """
        new = self.__class__
        return Iter((x, new(y)) for x, y in itertools.groupby(self._inner, key))
