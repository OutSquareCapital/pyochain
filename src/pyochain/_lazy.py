from __future__ import annotations

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
from dataclasses import dataclass
from functools import partial
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    NamedTuple,
    TypeIs,
    overload,
)

import cytoolz as cz
import more_itertools as mit

from ._iter import CommonMethods, convert_data
from ._option import Option
from ._result import Result

if TYPE_CHECKING:
    from ._eager import Seq, Set, Vec

type TryVal[T] = Option[T] | Result[T, object] | T | None
"""Represent a value that may be failible."""
type TryIter[T] = Iter[Option[T]] | Iter[Result[T, object]] | Iter[T | None]
"""Represent an iterator that may yield failible values."""
type Collector[T] = (
    Callable[[Iterable[T]], list[T]]
    | Callable[[Iterable[T]], tuple[T, ...]]
    | Callable[[Iterable[T]], set[T]]
    | Callable[[Iterable[T]], frozenset[T]]
)
"""Represent a function that collects an Iterable into a specific collection type."""
Position = Literal["first", "middle", "last", "only"]
"""Literal type representing the position of an item in an iterable."""


class Peeked[T](NamedTuple):
    values: tuple[T, ...]
    original: Iterator[T]


class Enumerated[T](NamedTuple):
    """Represents an item with its associated index in an enumeration."""

    idx: int
    """The index of the item in the enumeration."""
    value: T
    """The value of the item."""

    def __repr__(self) -> str:
        return f"({self.idx}, {self.value.__repr__()})"


@dataclass(slots=True)
class _CaseBuilder[T]:
    _iter: Iterable[T]
    _predicate: Callable[[T], bool]


@dataclass(slots=True)
class _WhenBuilder[T](_CaseBuilder[T]):
    def then[U](self, func: Callable[[T], U]) -> _ThenBuilder[T, U]:
        """Add a transformation to apply when the predicate is true.

        Args:
            func (Callable[[T], U]): Function to apply to items satisfying the predicate.

        Returns:
            _ThenBuilder[T, U]: Builder to chain further then() or finalize with or_else()/or_skip().
        """
        return _ThenBuilder(
            _iter=self._iter,
            _predicate=self._predicate,
            _func=func,
        )

    def or_else[U](self, func_else: Callable[[T], U]) -> Iter[T | U]:
        """Apply a function to items not satisfying the predicate.

        Args:
            func_else (Callable[[T], U]): Function to apply to items not satisfying the predicate.

        Returns:
            Iter[T | U]: An Iter with transformed items.
        """
        return Iter(
            item if self._predicate(item) else func_else(item) for item in self._iter
        )

    def or_skip(self) -> Iter[T]:
        """Skip items not satisfying the predicate.

        All items satisfying the predicate are retained.

        Returns:
            Iter[T]: An Iter with only items satisfying the predicate.

        """
        return Iter(item for item in self._iter if self._predicate(item))


@dataclass(slots=True)
class _ThenBuilder[T, R](_CaseBuilder[T]):
    _func: Callable[[T], R]

    def then[U](self, func: Callable[[R], U]) -> _ThenBuilder[T, U]:
        """Add a transformation to apply when the predicate is true.

        The function is composed with the result from the previous `then()`.

        Args:
            func (Callable[[R], U]): Function to apply to items satisfying the predicate.

        Returns:
            _ThenBuilder[T, U]: Builder to chain further then() or finalize with or_else()/or_skip().
        """
        return _ThenBuilder(
            _iter=self._iter,
            _predicate=self._predicate,
            _func=lambda x: func(self._func(x)),
        )

    def or_else[U](self, func_else: Callable[[T], U]) -> Iter[R | U]:
        """Apply a function to items not satisfying the predicate.

        Args:
            func_else (Callable[[T], U]): Function to apply to items not satisfying the predicate.

        Returns:
            Iter[R | U]: An Iter with transformed items.
        """
        return Iter(
            self._func(item) if self._predicate(item) else func_else(item)
            for item in self._iter
        )

    def or_skip(self) -> Iter[R]:
        """Skip items not satisfying the predicate.

        All items satisfying the predicate are retained.

        Returns:
            Iter[R]: An Iter with only items satisfying the predicate.

        """
        return Iter(self._func(item) for item in self._iter if self._predicate(item))


class Iter[T](CommonMethods[T], Iterator[T]):
    """A superset around Python's built-in `Iterator` Protocol, providing a rich set of functional programming tools.

    Implements the `Iterator` Protocol from `collections.abc`, so it can be used as a standard iterator.

    - An `Iterable` is any object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
    - An `Iterator` is an object representing a stream of data; returned by calling `iter()` on an `Iterable`.
    - Once an `Iterator` is exhausted, it cannot be reused or reset.

    It's designed around lazy evaluation, allowing for efficient processing of large datasets.

    - To instantiate from an `Iterable`, simply pass it to the standard constructor.
    - To instantiate from unpacked values, use the `from_` class method. This would be equivalent to the convenience of [x,y,z] syntax for lists.

    Once an `Iter` is created, it can be transformed and manipulated using a variety of chainable methods.

    However, keep in mind that `Iter` instances are single-use; once exhausted, they cannot be reused or reset.

    If you need to reuse the data, consider collecting it into a collection first with `.collect()`.

    You can always convert back to an `Iter` using `{Seq, Vec}.iter()` for free.

    In general, avoid intermediate references when dealing with lazy iterators, and prioritize method chaining instead.

    Args:
        data (Iterable[T]): Any object that can be iterated over.
    """

    _inner: Iterator[T]

    __slots__ = ("_inner",)

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = iter(data)  # pyright: ignore[reportIncompatibleVariableOverride]

    def __next__(self) -> T:
        return next(self._inner)

    def next(self) -> Option[T]:
        """Return the next element in the iterator.

        Note:
            The actual `.__next__()` method is conform to the Python `Iterator` Protocol, and is what will be actually called if you iterate over the `Iter` instance.

            `Iter.next()` is a convenience method that wraps the result in an `Option` to handle exhaustion gracefully, for custom use cases.

            Not only for typing, but for performance reasons, and coherence (iter(`Iter`) and iter(`Iter._inner`) wouldn't behave consistently otherwise).

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
        from ._option import Option

        return Option.from_(next(self, None))

    @staticmethod
    def from_count(start: int = 0, step: int = 1) -> Iter[int]:
        """Create an infinite `Iterator` of evenly spaced values.

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        Args:
            start (int): Starting value of the sequence. Defaults to 0.
            step (int): Difference between consecutive values. Defaults to 1.

        Returns:
            Iter[int]: An iterator generating the sequence.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_count(10, 2).take(3).collect()
        Seq(10, 12, 14)

        ```
        """
        return Iter(itertools.count(start, step))

    @overload
    @staticmethod
    def from_[U](data: Iterable[U]) -> Iter[U]: ...
    @overload
    @staticmethod
    def from_[U](data: U, *more_data: U) -> Iter[U]: ...
    @staticmethod
    def from_[U](data: Iterable[U] | U, *more_data: U) -> Iter[U]:
        """Create an iterator from any Iterable, or from unpacked values.

        Prefer using the standard constructor, as this method involves extra checks and conversions steps.

        Args:
            data (Iterable[U] | U): Iterable to convert into an iterator, or a single value.
            *more_data (U): Additional values to include if 'data' is not an Iterable.

        Returns:
            Iter[U]: A new Iter instance containing the provided data.

        Example:
        ```python
        >>> import pyochain as pc
        >>> # Creating from unpacked values
        >>> pc.Iter.from_(1, 2, 3).collect()
        Seq(1, 2, 3)

        ```
        """
        return Iter(convert_data(data, *more_data))

    @staticmethod
    def from_fn[S, V](
        state: S, generator: Callable[[S], Option[tuple[V, S]]]
    ) -> Iter[V]:
        """Create an `Iter` by repeatedly applying a **generator** function to an initial **state**.

        The **generator** function takes the current state and must return:

        - A tuple of `Some(value, new_state)` to emit the value `V` and continue with the new **state** `S`.
        - `NONE` to stop the generation.

        This is functionally equivalent to a state-based `while` loop.

        **Warning** ⚠️
            If the **generator** function never returns `NONE`, it creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken if necessary.

        Args:
            state (S): Initial state for the generator.
            generator (Callable[[S], Option[tuple[V, S]]]): Function that generates the next value and state.

        Returns:
            Iter[V]: An iterator generating values produced by the generator function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> # Example 1: Simple counter up to 5
        >>> def counter_generator(state: int) -> pc.Option[tuple[int, int]]:
        ...     if state < 5:
        ...         return pc.Some((state * 10, state + 1))
        ...     return pc.NONE
        >>> pc.Iter.from_fn(0, counter_generator).collect()
        Seq(0, 10, 20, 30, 40)
        >>> # Example 2: Fibonacci sequence up to 100
        >>> type FibState = tuple[int, int]
        >>> def fib_generator(state: FibState) -> pc.Option[tuple[int, FibState]]:
        ...     a, b = state
        ...     if a > 100:
        ...         return pc.NONE
        ...     return pc.Some((a, (b, a + b)))
        >>> pc.Iter.from_fn((0, 1), fib_generator).collect()
        Seq(0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
        >>> # Example 3: Infinite iterator (requires take())
        >>> pc.Iter.from_fn(1, lambda s: pc.Some((s, s * 2))).take(5).collect()
        Seq(1, 2, 4, 8, 16)

        ```
        """

        def _from_fn() -> Iterator[V]:
            current_state: S = state
            while True:
                result: Option[tuple[V, S]] = generator(current_state)
                if result.is_none():
                    break
                value, next_state = result.unwrap()
                yield value
                current_state = next_state

        return Iter(_from_fn())

    @overload
    def collect(self, collector: Callable[[Iterable[T]], list[T]]) -> Vec[T]: ...
    @overload
    def collect(
        self, collector: Callable[[Iterable[T]], tuple[T, ...]] = ...
    ) -> Seq[T]: ...
    @overload
    def collect(self, collector: Callable[[Iterable[T]], set[T]]) -> Set[T]: ...
    @overload
    def collect(self, collector: Callable[[Iterable[T]], frozenset[T]]) -> Set[T]: ...
    def collect(self, collector: Collector[T] = tuple) -> Seq[T] | Vec[T] | Set[T]:
        """Transforms an `Iter` into a collection.

        The most basic pattern in which collect() is used is to turn one collection into another.

        You take a collection, call `iter()` on it, do a bunch of transformations, and then `collect()` at the end.

        You can specify the target collection type by providing a **collector** function or type.

        This can be directly `list`, `tuple`, `frozenset` or `set`, or any `Callable` that takes an `Iterable[T]` and returns a `Collection[T]` of those types.

        Note:
            Use `.into()` if you want to convert the Iter into an unsupported container type directly (polars Series | DataFrame, pandas Series | DataFrame, numpy array,  etc.).

        Args:
            collector (Collector[T]): Function|type that defines the target collection type. Defaults to `tuple`, producing a `Seq[T]`.

        Returns:
            Seq[T] | Vec[T] | Set[T]: A materialized collection containing the collected elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(range(5)).collect()
        Seq(0, 1, 2, 3, 4)
        >>> data: tuple[int, ...] = (1, 2, 3)
        >>> iterator = pc.Iter.from_(data)
        >>> iterator._inner.__class__.__name__
        'tuple_iterator'
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped._inner.__class__.__name__
        'map'
        >>> mapped.collect()
        Seq(2, 4, 6)
        >>> # iterator is now exhausted
        >>> iterator.collect()
        Seq()
        >>> pc.Iter(range(5)).collect(list)
        Vec(0, 1, 2, 3, 4)
        >>> data: list[int] = [1, 2, 3]
        >>> iterator = pc.Iter.from_(data)
        >>> iterator._inner.__class__.__name__
        'list_iterator'

        ```
        """
        from ._eager import Seq, Set, Vec

        data = collector(self._inner)

        match data:
            case tuple():
                return Seq(data)
            case list():
                return Vec(data)
            case set() | frozenset():
                return Set(data)

    def try_collect[U](self: TryIter[U]) -> Option[Vec[U]]:
        """Fallibly transforms **self** into a `Vec`, short circuiting if a failure is encountered.

        `try_collect()` is a variation of `collect()` that allows fallible conversions during collection.

        Its main use case is simplifying conversions from iterators yielding `Option[T]`, `Result[T, E]` or `U | None` into `Option[Sequence[T]]`.

        Also, if a failure is encountered during `try_collect()`, the iterator is still valid and may continue to be used, in which case it will continue iterating starting after the element that triggered the failure.

        See the last example below for an example of how this works.

        Note:
            This method return `Vec[U]` instead of `Seq[U]` because the underlying data structure must be mutable in order to build up the collection.

        Returns:
            Option[Vec[U]]: `Some[Vec[U]]` if all elements were successfully collected, or `NONE` if a failure was encountered.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> # Successfully collecting an iterator of Option[int] into Option[Vec[int]]:
        >>> pc.Iter([pc.Some(1), pc.Some(2), pc.Some(3)]).try_collect()
        Some(Vec(1, 2, 3))
        >>> # Failing to collect in the same way:
        >>> pc.Iter([pc.Some(1), pc.Some(2), pc.NONE, pc.Some(3)]).try_collect()
        NONE
        >>> # A similar example, but with Result:
        >>> pc.Iter([pc.Ok(1), pc.Ok(2), pc.Ok(3)]).try_collect()
        Some(Vec(1, 2, 3))
        >>> pc.Iter([pc.Ok(1), pc.Err("error"), pc.Ok(3)]).try_collect()
        NONE
        >>> def external_fn(x: int) -> int | None:
        ...     if x % 2 == 0:
        ...         return x
        ...     return None
        >>> pc.Iter([1, 2, 3, 4]).map(external_fn).try_collect()
        NONE
        >>> # Demonstrating that the iterator remains usable after a failure:
        >>> it = pc.Iter([pc.Some(1), pc.NONE, pc.Some(3), pc.Some(4)])
        >>> it.try_collect()
        NONE
        >>> it.try_collect()
        Some(Vec(3, 4))

        ```
        """
        from ._eager import Vec
        from ._option import NONE, Some
        from ._result import Result

        def _try_collect(data: TryIter[U]) -> Option[Vec[U]]:
            collected = Vec[U].new()

            for item in data:
                if item is None:
                    return NONE
                match item:
                    case Result():
                        if item.is_err():
                            return NONE
                        collected.append(item.unwrap())  # pyright: ignore[reportUnknownArgumentType]
                    case Option():
                        if item.is_none():
                            return NONE
                        collected.append(item.unwrap())  # pyright: ignore[reportUnknownArgumentType]
                    case _ as plain_value:
                        collected.append(plain_value)
            return Some(collected)

        return self.into(_try_collect)

    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the Iterator by applying a function to each element in the iterable.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Args:
            func (Callable[Concatenate[T, P], Any]): Function to apply to each element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Returns:
            None: This is a terminal operation with no return value.


        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).iter().for_each(lambda x: print(x + 1))
        2
        3
        4

        ```
        """

        def _for_each(data: Iterable[T]) -> None:
            for v in data:
                func(v, *args, **kwargs)

        return self.into(_for_each)

    def array_chunks(self, size: int) -> Iter[Iter[T]]:
        """Yield subiterators (chunks) that each yield a fixed number elements, determined by size.

        The last chunk will be shorter if there are not enough elements.

        Args:
            size (int): Number of elements in each chunk.

        Returns:
            Iter[Iter[T]]: An iterable of iterators, each yielding n elements.

        If the sub-iterables are read in order, the elements of *iterable*
        won't be stored in memory.

        If they are read out of order, :func:`itertools.tee` is used to cache
        elements as necessary.
        ```python
        >>> import pyochain as pc
        >>> all_chunks = pc.Iter.from_count().array_chunks(4)
        >>> c_1, c_2, c_3 = all_chunks.next(), all_chunks.next(), all_chunks.next()
        >>> c_2.unwrap().collect()  # c_1's elements have been cached; c_3's haven't been
        Seq(4, 5, 6, 7)
        >>> c_1.unwrap().collect()
        Seq(0, 1, 2, 3)
        >>> c_3.unwrap().collect()
        Seq(8, 9, 10, 11)
        >>> pc.Seq([1, 2, 3, 4, 5, 6]).iter().array_chunks(3).map(lambda c: c.collect()).collect()
        Seq(Seq(1, 2, 3), Seq(4, 5, 6))
        >>> pc.Seq([1, 2, 3, 4, 5, 6, 7, 8]).iter().array_chunks(3).map(lambda c: c.collect()).collect()
        Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8))

        ```
        """
        from collections import deque
        from contextlib import suppress

        def _chunks(data: Iterable[T], size: int) -> Iterator[Iter[T]]:
            def _ichunk(
                iterator: Iterator[T], n: int
            ) -> tuple[Iterator[T], Callable[[int], int]]:
                cache: deque[T] = deque()
                chunk = itertools.islice(iterator, n)

                def generator() -> Iterator[T]:
                    with suppress(StopIteration):
                        while True:
                            if cache:
                                yield cache.popleft()
                            else:
                                yield next(chunk)

                def materialize_next(n: int) -> int:
                    to_cache = n - len(cache)

                    # materialize up to n
                    if to_cache > 0:
                        cache.extend(itertools.islice(chunk, to_cache))

                    # return number materialized up to n
                    return min(n, len(cache))

                return (generator(), materialize_next)

            iterator = iter(data)
            while True:
                # Create new chunk
                chunk, materialize_next = _ichunk(iterator, size)

                # Check to see whether we're at the end of the source iterable
                if not materialize_next(size):
                    return

                yield self.__class__(chunk)
                materialize_next(size)

        return self._iter(_chunks, size)

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
    def flatten(self: Iter[range]) -> Iter[int]: ...
    def flatten[U: Iterable[Any]](self: Iter[U]) -> Iter[Any]:
        """Flatten one level of nesting and return a new Iterable wrapper.

        This is a shortcut for `.apply(itertools.chain.from_iterable)`.

        Returns:
            Iter[Any]: An iterable of flattened elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([[1, 2], [3]]).flatten().collect()
        Seq(1, 2, 3)

        ```
        """
        return self._iter(itertools.chain.from_iterable)

    @overload
    def flat_map[U, R](
        self: Iter[Iterable[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[Iterator[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[Collection[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[KeysView[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[ValuesView[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[Iter[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[Seq[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[Set[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...

    @overload
    def flat_map[U, R](
        self: Iter[Sequence[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[list[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: Iter[tuple[U, ...]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[R](self: Iter[range], func: Callable[[int], R]) -> Iter[R]: ...
    def flat_map[U: Iterable[Any], R](
        self: Iter[U],
        func: Callable[[Any], R],
    ) -> Iter[Any]:
        """Map each element through func and flatten the result by one level.

        Args:
            func (Callable[[Any], R]): Function to apply to each element.

        Returns:
            Iter[Any]: An iterable of flattened transformed elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = [[1, 2], [3, 4]]
        >>> pc.Seq(data).iter().flat_map(lambda x: x + 10).collect()
        Seq(11, 12, 13, 14)

        ```
        """

        def _flat_map(data: Iterable[U]) -> map[R]:
            return map(func, itertools.chain.from_iterable(data))

        return self._iter(_flat_map)

    def unique_to_each[U: Iterable[Any]](self: Iter[U]) -> Iter[Iter[U]]:
        """Return the elements from each of the iterators that aren't in the other iterators.

        It is assumed that the elements of each iterable are hashable.

        **Credits**

            more_itertools.unique_to_each

        Returns:
            Iter[Iter[U]]: An iterator of iterators, each containing the unique elements from the corresponding input iterable.

        For example, suppose you have a set of packages, each with a set of dependencies:

        **{'pkg_1': {'A', 'B'}, 'pkg_2': {'B', 'C'}, 'pkg_3': {'B', 'D'}}**

        If you remove one package, which dependencies can also be removed?

        If pkg_1 is removed, then A is no longer necessary - it is not associated with pkg_2 or pkg_3.

        Similarly, C is only needed for pkg_2, and D is only needed for pkg_3:

        ```python
        >>> import pyochain as pc
        >>> data = ({"A", "B"}, {"B", "C"}, {"B", "D"})
        >>> pc.Iter(data).unique_to_each().map(lambda x: x.into(list)).collect()
        Seq(['A'], ['C'], ['D'])

        ```

        If there are duplicates in one input iterable that aren't in the others they will be duplicated in the output.

        Input order is preserved:
        ```python
        >>> data = ("mississippi", "missouri")
        >>> pc.Seq(data).iter().unique_to_each().map(lambda x: x.into(list)).collect()
        Seq(['p', 'p'], ['o', 'u', 'r'])

        ```
        """
        from collections import Counter

        def _unique_to_each(data: Iterable[U]) -> Iterator[Iter[U]]:
            pool: tuple[Iterable[U], ...] = tuple(data)
            counts: Counter[U] = Counter(itertools.chain.from_iterable(map(set, pool)))
            uniques: set[U] = {element for element in counts if counts[element] == 1}
            return ((Iter(filter(uniques.__contains__, it))) for it in pool)

        return self._iter(_unique_to_each)

    def split_into(self, *sizes: Option[int]) -> Iter[Iter[T]]:
        """Yield a list of sequential items from iterable of length 'n' for each integer 'n' in sizes.

        Args:
            *sizes (Option[int]): `Some` integers specifying the sizes of each chunk. Use `NONE` for the remainder.

        Returns:
            Iter[Iter[T]]: An iterator of iterators, each containing a chunk of the original iterable.

        If the sum of sizes is smaller than the length of iterable, then the remaining items of iterable will not be returned.

        If the sum of sizes is larger than the length of iterable:

        - fewer items will be returned in the iteration that overruns the iterable
        - further lists will be empty

        When a `NONE` object is encountered in sizes, the returned list will contain items up to the end of iterable the same way that itertools.slice does.

        split_into can be useful for grouping a series of items where the sizes of the groups are not uniform.

        An example would be where in a row from a table:

        - multiple columns represent elements of the same feature (e.g. a point represented by x,y,z)
        - the format is not the same for all columns.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def _get_results(x: pc.Iter[pc.Iter[int]]) -> pc.Seq[pc.Seq[int]]:
        ...    return x.map(lambda x: x.collect()).collect()
        >>>
        >>> data = [1, 2, 3, 4, 5, 6]
        >>> pc.Iter(data).split_into(pc.Some(1), pc.Some(2), pc.Some(3)).into(_get_results)
        Seq(Seq(1,), Seq(2, 3), Seq(4, 5, 6))
        >>> pc.Iter(data).split_into(pc.Some(2), pc.Some(3)).into(_get_results)
        Seq(Seq(1, 2), Seq(3, 4, 5))
        >>> pc.Iter([1, 2, 3, 4]).split_into(pc.Some(1), pc.Some(2), pc.Some(3), pc.Some(4)).into(_get_results)
        Seq(Seq(1,), Seq(2, 3), Seq(4,), Seq())
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        >>> pc.Iter(data).split_into(pc.Some(2), pc.Some(3), pc.NONE).into(_get_results)
        Seq(Seq(1, 2), Seq(3, 4, 5), Seq(6, 7, 8, 9, 0))

        ```
        """

        def _split_into(data: Iterable[T]) -> Iterator[Iter[T]]:
            """Credits: more_itertools.split_into."""
            it = iter(data)

            for size in sizes:
                if size.is_none():
                    yield self.__class__(it)
                    return
                else:
                    yield self.__class__(itertools.islice(it, size.unwrap()))

        return self._iter(_split_into)

    def split_when(
        self,
        predicate: Callable[[T, T], bool],
        max_split: int = -1,
    ) -> Iter[Iter[T]]:
        """Split iterable into pieces based on the output of a predicate function.

        Args:
            predicate (Callable[[T, T], bool]): Function that takes successive pairs of items and returns True if the iterable should be split.
            max_split (int): Maximum number of splits to perform. Defaults to -1 (no limit).

        Returns:
            Iter[Iter[T]]: An iterator of iterators of items.

        At most *max_split* splits are done.

        If *max_split* is not specified or -1, then there is no limit on the number of splits.

        The example below shows how to find runs of increasing numbers, by splitting the iterable when element i is larger than element i + 1.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq([1, 2, 3, 3, 2, 5, 2, 4, 2])
        >>> data.iter().split_when(lambda x, y: x > y).map(lambda x: x.collect()).collect()
        Seq(Seq(1, 2, 3, 3), Seq(2, 5), Seq(2, 4), Seq(2,))
        >>> data.iter().split_when(lambda x, y: x > y, max_split=2).map(lambda x: x.collect()).collect()
        Seq(Seq(1, 2, 3, 3), Seq(2, 5), Seq(2, 4, 2))

        ```
        """

        def _split_when(data: Iterable[T], max_split: int) -> Iterator[Iter[T]]:
            """Credits: more_itertools.split_when."""
            if max_split == 0:
                yield self
                return

            it = iter(data)
            try:
                cur_item = next(it)
            except StopIteration:
                return

            buf = [cur_item]
            for next_item in it:
                if predicate(cur_item, next_item):
                    yield Iter(buf)
                    if max_split == 1:
                        yield Iter((next_item, *it))
                        return
                    buf = []
                    max_split -= 1

                buf.append(next_item)
                cur_item = next_item

            yield Iter(buf)

        return self._iter(_split_when, max_split)

    def split_at(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
        *,
        keep_separator: bool = False,
    ) -> Iter[Iter[T]]:
        """Yield iterators of items from iterable, where each iterator is delimited by an item where `predicate` returns True.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform. Defaults to -1 (no limit).
            keep_separator (bool): Whether to include the separator in the output. Defaults to False.

        Returns:
            Iter[Iter[T]]: An iterator of iterators, each containing a segment of the original iterable.

        By default, the delimiting items are not included in the output.

        To include them, set *keep_separator* to `True`.
        At most *max_split* splits are done.

        If *max_split* is not specified or -1, then there is no limit on the number of splits.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def _to_res(x: pc.Iter[pc.Iter[str]]) -> pc.Seq[pc.Seq[str]]:
        ...     return x.map(lambda x: x.into(list)).collect()
        >>>
        >>> pc.Iter("abcdcba").split_at(lambda x: x == "b").into(_to_res)
        Seq(['a'], ['c', 'd', 'c'], ['a'])
        >>> pc.Iter(range(10)).split_at(lambda n: n % 2 == 1).into(_to_res)
        Seq([0], [2], [4], [6], [8], [])
        >>> pc.Iter(range(10)).split_at(lambda n: n % 2 == 1, max_split=2).into(_to_res)
        Seq([0], [2], [4, 5, 6, 7, 8, 9])
        >>>
        >>> def cond(x: str) -> bool:
        ...     return x == "b"
        >>>
        >>> pc.Iter("abcdcba").split_at(cond, keep_separator=True).into(_to_res)
        Seq(['a'], ['b'], ['c', 'd', 'c'], ['b'], ['a'])

        ```
        """

        def _split_at(data: Iterable[T], max_split: int) -> Iterator[Iter[T]]:
            """Credits: more_itertools.split_at."""
            if max_split == 0:
                yield self
                return

            buf: list[T] = []
            it = iter(data)
            for item in it:
                if predicate(item):
                    yield self.__class__(buf)
                    if keep_separator:
                        yield self.__class__((item,))
                    if max_split == 1:
                        yield self.__class__(it)
                        return
                    buf = []
                    max_split -= 1
                else:
                    buf.append(item)
            yield self.__class__(buf)

        return self._iter(_split_at, max_split)

    def split_after(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
    ) -> Iter[Iter[T]]:
        """Yield iterator of items from iterable, where each iterator ends with an item where `predicate` returns True.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform. Defaults to -1 (no limit).

        Returns:
            Iter[Iter[T]]: An iterable of lists of items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("one1two2").split_after(str.isdigit).map(list).collect()
        Seq(['o', 'n', 'e', '1'], ['t', 'w', 'o', '2'])

        >>> def cond(n: int) -> bool:
        ...     return n % 3 == 0
        >>>
        >>> pc.Iter(range(10)).split_after(cond).map(list).collect()
        Seq([0], [1, 2, 3], [4, 5, 6], [7, 8, 9])
        >>> pc.Iter(range(10)).split_after(cond, max_split=2).map(list).collect()
        Seq([0], [1, 2, 3], [4, 5, 6, 7, 8, 9])

        ```
        """

        def _split_after(data: Iterable[T], max_split: int) -> Iterator[Iter[T]]:
            """Credits: more_itertools.split_after."""
            if max_split == 0:
                yield self.__class__(data)
                return

            buf: list[T] = []
            it = iter(data)
            for item in it:
                buf.append(item)
                if predicate(item) and buf:
                    yield self.__class__(buf)
                    if max_split == 1:
                        buf = list(it)
                        if buf:
                            yield self.__class__(buf)
                        return
                    buf = []
                    max_split -= 1
            if buf:
                yield self.__class__(buf)

        return self._iter(_split_after, max_split)

    def split_before(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
    ) -> Iter[Iter[T]]:
        """Yield iterator of items from iterable, where each iterator ends with an item where `predicate` returns True.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform. Defaults to -1 (no limit).

        Returns:
            Iter[Iter[T]]: An iterable of lists of items.


        At most *max_split* are done.


        If *max_split* is not specified or -1, then there is no limit on the number of splits:

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("abcdcba").split_before(lambda x: x == "b").map(list).collect()
        Seq(['a'], ['b', 'c', 'd', 'c'], ['b', 'a'])
        >>>
        >>> def cond(n: int) -> bool:
        ...     return n % 2 == 1
        >>>
        >>> pc.Iter(range(10)).split_before(cond).map(list).collect()
        Seq([0], [1, 2], [3, 4], [5, 6], [7, 8], [9])
        >>> pc.Iter(range(10)).split_before(cond, max_split=2).map(list).collect()
        Seq([0], [1, 2], [3, 4, 5, 6, 7, 8, 9])

        ```
        """

        def _split_before(data: Iterable[T], max_split: int) -> Iterator[Iter[T]]:
            """Credits: more_itertools.split_before."""
            if max_split == 0:
                yield self.__class__(data)
                return

            buf: list[T] = []
            it = iter(data)
            for item in it:
                if predicate(item) and buf:
                    yield self.__class__(buf)
                    if max_split == 1:
                        yield self.__class__([item, *it])
                        return
                    buf = []
                    max_split -= 1
                buf.append(item)
            if buf:
                yield self.__class__(buf)

        return self._iter(_split_before, max_split)

    def find_map[R](self, func: Callable[[T], Option[R]]) -> Option[R]:
        """Applies function to the elements of the `Iterator` and returns the first Some(R) result.

        `Iter.find_map(f)` is equivalent to `Iter.filter_map(f).next()`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element, returning an `Option[R]`.

        Returns:
            Option[R]: The first `Some(R)` result from applying `func`, or `NONE` if no such result is found.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> def _parse(s: str) -> pc.Option[int]:
        ...     try:
        ...         return pc.Some(int(s))
        ...     except ValueError:
        ...         return pc.NONE
        >>>
        >>> pc.Iter(["lol", "NaN", "2", "5"]).find_map(_parse)
        Some(2)

        ```
        """
        return self.filter_map(func).next()

    # map -----------------------------------------------------------------

    def map[R](self, func: Callable[[T], R]) -> Iter[R]:
        """Apply a function to each element of the iterable.

        If you are good at thinking in types, you can think of map() like this:
            If you have an iterator that gives you elements of some type A, and you want an iterator of some other type B, you can use map(),
            passing a closure that takes an A and returns a B.

        map() is conceptually similar to a for loop.

        However, as map() is lazy, it is best used when you are already working with other iterators.

        If you are doing some sort of looping for a side effect, it is considered more idiomatic to use `for_each` than map().

        Args:
            func (Callable[[T], R]): Function to apply to each element.

        Returns:
            Iter[R]: An iterator of transformed elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).map(lambda x: x + 1).collect()
        Seq(2, 3)

        ```
        """
        return self._iter(partial(map, func))

    def map_star[U: Iterable[Any], R](
        self: Iter[U],
        func: Callable[..., R],
    ) -> Iter[R]:
        """Applies a function to each element.where each element is an iterable.

        Unlike `.map()`, which passes each element as a single argument,
        `.starmap()` unpacks each element into positional arguments for the function.

        In short, for each `element` in the sequence, it computes `func(*element)`.

        - Use map_star when the performance matters (it is faster).
        - Use map with unpacking when readability matters (the types can be inferred).

        Args:
            func (Callable[..., R]): Function to apply to unpacked elements.

        Returns:
            Iter[R]: An iterable of results from applying the function to unpacked elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def make_sku(color, size):
        ...     return f"{color}-{size}"
        >>> data = pc.Seq(["blue", "red"])
        >>> data.iter().product(["S", "M"]).map_star(make_sku).collect()
        Seq('blue-S', 'blue-M', 'red-S', 'red-M')
        >>> # This is equivalent to:
        >>> data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).collect()
        Seq('blue-S', 'blue-M', 'red-S', 'red-M')

        ```
        """
        return self._iter(partial(itertools.starmap, func))

    def map_if(self, predicate: Callable[[T], bool]) -> _WhenBuilder[T]:
        """Begin a conditional transformation chain on an Iter.

        Args:
            predicate (Callable[[T], bool]): Function to test each item.

        Returns:
            _WhenBuilder[T]: Builder to chain then() and or_else()/or_skip() calls.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq(range(-3, 4))
        >>> data.iter().map_if(lambda x: x > 0).then(lambda x: x * 10).or_else(lambda x: x).collect()
        Seq(-3, -2, -1, 0, 10, 20, 30)
        >>> data.iter().map_if(lambda x: x % 2 == 0).then(lambda x: f"{x} is even").or_skip().collect()
        Seq('-2 is even', '0 is even', '2 is even')

        ```
        """
        return self.into(
            _WhenBuilder,
            _predicate=predicate,
        )

    def repeat(
        self,
        n: int,
        factory: Callable[[Iterable[T]], Sequence[T]] = tuple,
    ) -> Iter[Iterable[T]]:
        """Repeat the entire iterable n times (as elements).

        Args:
            n (int): Number of repetitions.
            factory (Callable[[Iterable[T]], Sequence[T]]): Factory to create the repeated Sequence (default: tuple).

        Returns:
            Iter[Iterable[T]]: An iterable of repeated sequences.
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).repeat(2).collect()
        Seq((1, 2), (1, 2))
        >>> pc.Iter([1, 2]).repeat(3, list).collect()
        Seq([1, 2], [1, 2], [1, 2])

        ```
        """

        def _repeat(data: Iterable[T]) -> Iterator[Iterable[T]]:
            return itertools.repeat(factory(data), n)

        return self._iter(_repeat)

    def scan[U](self, state: U, func: Callable[[U, T], Option[U]]) -> Iter[U]:
        """Transform elements by sharing state between iterations.

        `scan` takes two arguments:
            - an initial value which seeds the internal state
            - a closure with two arguments

        The first being a reference to the internal state and the second an iterator element.

        The closure can assign to the internal state to share state between iterations.

        On iteration, the closure will be applied to each element of the iterator and the return value from the closure, an Option, is returned by the next method.

        Thus the closure can return Some(value) to yield value, or None to end the iteration.

        Args:
            state (U): Initial state.
            func (Callable[[U, T], Option[U]]): Function that takes the current state and an item, and returns an Option.

        Returns:
            Iter[U]: An iterable of the yielded values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def accumulate_until_limit(state: int, item: int) -> pc.Option[int]:
        ...     new_state = state + item
        ...     match new_state:
        ...         case _ if new_state <= 10:
        ...             return pc.Some(new_state)
        ...         case _:
        ...             return pc.NONE
        >>> pc.Iter([1, 2, 3, 4, 5]).scan(0, accumulate_until_limit).collect()
        Seq(1, 3, 6, 10)

        ```
        """

        def gen(data: Iterable[T]) -> Iterator[U]:
            current: U = state
            for item in data:
                res = func(current, item)
                if res.is_none():
                    break
                current = res.unwrap()
                yield res.unwrap()

        return self._iter(gen)

    # filters ------------------------------------------------------------
    @overload
    def filter[U](self, func: Callable[[T], TypeIs[U]]) -> Iter[U]: ...
    @overload
    def filter(self, func: Callable[[T], bool]) -> Iter[T]: ...
    def filter[U](self, func: Callable[[T], bool | TypeIs[U]]) -> Iter[T] | Iter[U]:
        """Creates an `Iter` which uses a closure to determine if an element should be yielded.

        Given an element the closure must return true or false.

        The returned `Iter` will yield only the elements for which the closure returns true.

        The closure can return a `TypeIs` to narrow the type of the returned iterable.

        This won't have any runtime effect, but allows for better type inference.

        Note:
            `Iter.filter(f).next()` is equivalent to `Iter.find(f)`.

        Args:
            func (Callable[[T], bool | TypeIs[U]]): Function to evaluate each item.

        Returns:
            Iter[T] | Iter[U]: An iterable of the items that satisfy the predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = (1, 2, 3)
        >>> pc.Iter(data).filter(lambda x: x > 1).collect()
        Seq(2, 3)
        >>> pc.Iter(data).filter(lambda x: x > 1).next()
        Some(2)
        >>> pc.Iter(data).find(lambda x: x > 1)
        Some(2)

        ```
        """

        def _filter(data: Iterable[T]) -> Iterator[T]:
            return filter(func, data)

        return self._iter(_filter)

    def filter_false(self, func: Callable[[T], bool]) -> Iter[T]:
        """Return elements for which func is false.

        Args:
            func (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Iter[T]: An iterable of the items that do not satisfy the predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).filter_false(lambda x: x > 1).collect()
        Seq(1,)

        ```
        """
        return self._iter(partial(itertools.filterfalse, func))

    def take_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Take items while predicate holds.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Iter[T]: An iterable of the items taken while the predicate is true.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 0)).take_while(lambda x: x > 0).collect()
        Seq(1, 2)

        ```
        """
        return self._iter(partial(itertools.takewhile, predicate))

    def skip_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Drop items while predicate holds.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Iter[T]: An iterable of the items after skipping those for which the predicate is true.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 0)).skip_while(lambda x: x > 0).collect()
        Seq(0,)

        ```
        """
        return self._iter(partial(itertools.dropwhile, predicate))

    def compress(self, *selectors: bool) -> Iter[T]:
        """Filter elements using a boolean selector iterable.

        Args:
            *selectors (bool): Boolean values indicating which elements to keep.

        Returns:
            Iter[T]: An iterable of the items selected by the boolean selectors.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCDEF").compress(1, 0, 1, 0, 1, 1).collect()
        Seq('A', 'C', 'E', 'F')

        ```
        """
        return self._iter(itertools.compress, selectors)

    def unique(self, key: Callable[[T], Any] | None = None) -> Iter[T]:
        """Return only unique elements of the iterable.

        Args:
            key (Callable[[T], Any] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            Iter[T]: An iterable of the unique items.

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
        return self._iter(cz.itertoolz.unique, key=key)

    def take(self, n: int) -> Iter[T]:
        """Creates an iterator that yields the first n elements, or fewer if the underlying iterator ends sooner.

        `Iter.take(n)` yields elements until n elements are yielded or the end of the iterator is reached (whichever happens first).

        The returned iterator is either:

        - A prefix of length n if the original iterator contains at least n elements
        - All of the (fewer than n) elements of the original iterator if it contains fewer than n elements.

        Args:
            n (int): Number of elements to take.

        Returns:
            Iter[T]: An iterable of the first n items.

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
        return self._iter(partial(cz.itertoolz.take, n))

    def skip(self, n: int) -> Iter[T]:
        """Drop first n elements.

        Args:
            n (int): Number of elements to skip.

        Returns:
            Iter[T]: An iterable of the items after skipping the first n items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 3)).skip(1).collect()
        Seq(2, 3)

        ```
        """
        return self._iter(partial(cz.itertoolz.drop, n))

    def unique_justseen(self, key: Callable[[T], Any] | None = None) -> Iter[T]:
        """Yields elements in order, ignoring serial duplicates.

        Args:
            key (Callable[[T], Any] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            Iter[T]: An iterable of the unique items, preserving order.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("AAAABBBCCDAABBB").unique_justseen().collect()
        Seq('A', 'B', 'C', 'D', 'A', 'B')
        >>> pc.Iter.from_("ABBCcAD").unique_justseen(str.lower).collect()
        Seq('A', 'B', 'C', 'A', 'D')

        ```
        """
        return self._iter(mit.unique_justseen, key=key)

    def unique_in_window(
        self,
        n: int,
        key: Callable[[T], Any] | None = None,
    ) -> Iter[T]:
        """Yield the items from iterable that haven't been seen recently.

        The items in iterable must be hashable.

        Args:
            n (int): Size of the lookback window.
            key (Callable[[T], Any] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            Iter[T]: An iterable of the items that are unique within the specified window.

        Example:
        ```python
        >>> import pyochain as pc
        >>> iterable = [0, 1, 0, 2, 3, 0]
        >>> n = 3
        >>> pc.Iter(iterable).unique_in_window(n).collect()
        Seq(0, 1, 2, 3, 0)

        ```
        The key function, if provided, will be used to determine uniqueness:
        ```python
        >>> pc.Iter.from_("abAcda").unique_in_window(3, key=str.lower).collect()
        Seq('a', 'b', 'c', 'd', 'a')

        ```
        """
        return self._iter(mit.unique_in_window, n, key=key)

    def extract(self, indices: Iterable[int]) -> Iter[T]:
        """Yield values at the specified indices.

        - The iterable is consumed lazily and can be infinite.
        - The indices are consumed immediately and must be finite.
        - Raises IndexError if an index lies beyond the iterable.
        - Raises ValueError for negative indices.

        Args:
            indices (Iterable[int]): Iterable of indices to extract values from.

        Returns:
            Iter[T]: An iterable of the extracted items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> text = "abcdefghijklmnopqrstuvwxyz"
        >>> pc.Iter(text).extract([7, 4, 11, 11, 14]).collect()
        Seq('h', 'e', 'l', 'l', 'o')

        ```
        """
        return self._iter(mit.extract, indices)

    def step_by(self, step: int) -> Iter[T]:
        """Creates an `Iter` starting at the same point, but stepping by the given **step** at each iteration.

        Note:
            The first element of the iterator will always be returned, regardless of the **step** given.

        Args:
            step (int): Step size for selecting items.

        Returns:
            Iter[T]: An iterable of every nth item.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([0, 1, 2, 3, 4, 5]).step_by(2).collect()
        Seq(0, 2, 4)

        ```
        """
        return self._iter(partial(cz.itertoolz.take_nth, step))

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Iter[T]:
        """Return a slice of the iterable.

        Args:
            start (int | None): Starting index of the slice. Defaults to None.
            stop (int | None): Ending index of the slice. Defaults to None.
            step (int | None): Step size for the slice. Defaults to None.

        Returns:
            Iter[T]: An iterable of the sliced items.

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

        def _slice(data: Iterable[T]) -> Iterator[T]:
            return itertools.islice(data, start, stop, step)

        return self._iter(_slice)

    def filter_map[R](self, func: Callable[[T], Option[R]]) -> Iter[R]:
        """Creates an iterator that both filters and maps.

        The returned iterator yields only the values for which the supplied closure returns Some(value).

        `filter_map` can be used to make chains of `filter` and map more concise.

        The example below shows how a `map().filter().map()` can be shortened to a single call to `filter_map`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each item.

        Returns:
            Iter[R]: An iterable of the results where func returned `Some`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def _parse(s: str) -> pc.Result[int, str]:
        ...     try:
        ...         return pc.Ok(int(s))
        ...     except ValueError:
        ...         return pc.Err(f"Invalid integer, got {s!r}")
        >>>
        >>> data = pc.Seq(["1", "two", "NaN", "four", "5"])
        >>> data.iter().filter_map(lambda s: _parse(s).ok()).collect()
        Seq(1, 5)
        >>> # Equivalent to:
        >>> (
        ...     data.iter()
        ...    .map(lambda s: _parse(s).ok())
        ...    .filter(lambda s: s.is_some())
        ...    .map(lambda s: s.unwrap())
        ...    .collect()
        ... )
        Seq(1, 5)

        ```
        """

        def _filter_map(data: Iterable[T]) -> Iterator[R]:
            for item in data:
                res = func(item)
                if res.is_some():
                    yield res.unwrap()

        return self._iter(_filter_map)

    # joins and zips ------------------------------------------------------------
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
    def zip(
        self,
        *others: Iterable[Any],
        strict: bool = False,
    ) -> Iter[tuple[Any, ...]]:
        """Yields n-length tuples, where n is the number of iterables passed as positional arguments.

        The i-th element in every tuple comes from the i-th iterable argument to `.zip()`.

        This continues until the shortest argument is exhausted.

        Args:
            *others (Iterable[Any]): Other iterables to zip with.
            strict (bool): If `True` and one of the arguments is exhausted before the others, raise a ValueError. Defaults to `False`.

        Returns:
            Iter[tuple[Any, ...]]: An `Iter` of tuples containing elements from the zipped Iter and other iterables.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).zip([10, 20]).collect()
        Seq((1, 10), (2, 20))
        >>> pc.Iter(["a", "b"]).zip([1, 2, 3]).collect()
        Seq(('a', 1), ('b', 2))

        ```
        """
        return self._iter(zip, *others, strict=strict)

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
        *iterables: Iterable[T],
    ) -> Iter[tuple[Option[T], ...]]: ...
    def zip_longest(self, *others: Iterable[Any]) -> Iter[tuple[Option[Any], ...]]:
        """Return a zip Iterator who yield a tuple where the i-th element comes from the i-th iterable argument.

        Yield values until the longest iterable in the argument sequence is exhausted, and then it raises StopIteration.

        The longest iterable determines the length of the returned iterator, and will return `Some[T]` until exhaustion.

        When the shorter iterables are exhausted, they yield `NONE`.

        Args:
            *others (Iterable[Any]): Other iterables to zip with.

        Returns:
            Iter[tuple[Option[Any], ...]]: An iterable of tuples containing optional elements from the zipped iterables.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).zip_longest([10]).collect()
        Seq((Some(1), Some(10)), (Some(2), NONE))
        >>> # Can be combined with try collect to filter out the NONE:
        >>> pc.Iter([1, 2]).zip_longest([10]).map(lambda x: pc.Iter(x).try_collect()).collect()
        Seq(Some(Vec(1, 10)), NONE)

        ```
        """
        from ._option import Option

        def _zip_longest(data: Iterable[T]) -> Iterator[tuple[Option[T], ...]]:
            return (
                tuple(Option.from_(t) for t in tup)
                for tup in itertools.zip_longest(data, *others, fillvalue=None)
            )

        return self._iter(_zip_longest)

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

    def product(self, *others: Iterable[Any]) -> Iter[tuple[Any, ...]]:
        """Computes the Cartesian product with another iterable.

        This is the declarative equivalent of nested for-loops.

        It pairs every element from the source iterable with every element from the
        other iterable.

        Args:
            *others (Iterable[Any]): Other iterables to compute the Cartesian product with.

        Returns:
            Iter[tuple[Any, ...]]: An iterable of tuples containing elements from the Cartesian product.

        Example:
        ```python
        >>> import pyochain as pc
        >>> sizes = ["S", "M"]
        >>> pc.Iter(["blue", "red"]).product(sizes).collect()
        Seq(('blue', 'S'), ('blue', 'M'), ('red', 'S'), ('red', 'M'))

        ```
        """
        return self._iter(itertools.product, *others)

    def diff_at(
        self,
        *others: Iterable[T],
        default: T | None = None,
        key: Callable[[T], Any] | None = None,
    ) -> Iter[tuple[T, ...]]:
        """Return those items that differ between iterables.

        Each output item is a tuple where the i-th element is from the i-th input iterable.

        If an input iterable is exhausted before others, then the corresponding output items will be filled with *default*.

        Args:
            *others (Iterable[T]): Other iterables to compare with.
            default (T | None): Value to use for missing elements. Defaults to None.
            key (Callable[[T], Any] | None): Function to apply to each item for comparison. Defaults to None.

        Returns:
            Iter[tuple[T, ...]]: An iterable of tuples containing differing elements from the input iterables.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq([1, 2, 3])
        >>> data.iter().diff_at([1, 2, 10, 100], default=None).collect()
        Seq((3, 10), (None, 100))
        >>> data.iter().diff_at([1, 2, 10, 100, 2, 6, 7], default=0).collect()
        Seq((3, 10), (0, 100), (0, 2), (0, 6), (0, 7))

        A key function may also be applied to each item to use during comparisons:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(["apples", "bananas"]).diff_at(["Apples", "Oranges"], key=str.lower).collect()
        Seq(('bananas', 'Oranges'),)

        ```
        """
        return self._iter(cz.itertoolz.diff, *others, default=default, key=key)

    def join_with[R, K](
        self,
        other: Iterable[R],
        left_on: Callable[[T], K],
        right_on: Callable[[R], K],
        left_default: T | None = None,
        right_default: R | None = None,
    ) -> Iter[tuple[T, R]]:
        """Perform a relational join with another iterable.

        Args:
            other (Iterable[R]): Iterable to join with.
            left_on (Callable[[T], K]): Function to extract the join key from the left iterable.
            right_on (Callable[[R], K]): Function to extract the join key from the right iterable.
            left_default (T | None): Default value for missing elements in the left iterable. Defaults to None.
            right_default (R | None): Default value for missing elements in the right iterable. Defaults to None.

        Returns:
            Iter[tuple[T, R]]: An iterator yielding tuples of joined elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> sizes = ["S", "M"]
        >>> pc.Iter(["blue", "red"]).join_with(sizes, left_on=lambda c: c, right_on=lambda s: s).collect()
        Seq((None, 'S'), (None, 'M'), ('blue', None), ('red', None))

        ```
        """

        def _join(data: Iterable[T]) -> Iterator[tuple[T, R]]:
            return cz.itertoolz.join(
                leftkey=left_on,
                leftseq=data,
                rightkey=right_on,
                rightseq=other,
                left_default=left_default,
                right_default=right_default,
            )

        return self._iter(_join)

    # windows and partitions ------------------------------------------------------------
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
        self, length: int, func: Callable[[tuple[T, ...]], R]
    ) -> Iter[R]: ...
    def map_windows[R](
        self, length: int, func: Callable[[tuple[Any, ...]], R]
    ) -> Iter[R]:
        """Calls the given **func** for each contiguous window of size **length** over **self**.

        The windows during mapping overlaps.

        The provided function must have a signature matching the length of the window.

        Args:
            length (int): The length of each window.
            func (Callable[[tuple[Any, ...]], R]): Function to apply to each window.

        Returns:
            Iter[R]: An iterator over the outputs of func.

        Example:
        ```python
        >>> import pyochain as pc

        >>> pc.Iter("abcd").map_windows(2, lambda xy: f"{xy[0]}+{xy[1]}").collect()
        Seq('a+b', 'b+c', 'c+d')
        >>> pc.Iter([1, 2, 3, 4]).map_windows(2, lambda xy: xy).collect()
        Seq((1, 2), (2, 3), (3, 4))
        >>> def moving_average(seq: tuple[int, ...]) -> float:
        ...     return float(sum(seq)) / len(seq)
        >>> pc.Iter([1, 2, 3, 4]).map_windows(2, moving_average).collect()
        Seq(1.5, 2.5, 3.5)

        ```
        """

        def _map_windows(data: Iterable[T]) -> Iterator[R]:
            return map(func, cz.itertoolz.sliding_window(length, data))

        return self._iter(_map_windows)

    @overload
    def partition(self, n: Literal[1], pad: None = None) -> Iter[tuple[T]]: ...
    @overload
    def partition(self, n: Literal[2], pad: None = None) -> Iter[tuple[T, T]]: ...
    @overload
    def partition(self, n: Literal[3], pad: None = None) -> Iter[tuple[T, T, T]]: ...
    @overload
    def partition(self, n: Literal[4], pad: None = None) -> Iter[tuple[T, T, T, T]]: ...
    @overload
    def partition(
        self,
        n: Literal[5],
        pad: None = None,
    ) -> Iter[tuple[T, T, T, T, T]]: ...
    @overload
    def partition(self, n: int, pad: int) -> Iter[tuple[T, ...]]: ...
    def partition(self, n: int, pad: int | None = None) -> Iter[tuple[T, ...]]:
        """Partition sequence into tuples of length n.

        Args:
            n (int): Length of each partition.
            pad (int | None): Value to pad the last partition if needed. Defaults to None.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3, 4]).partition(2).collect()
        Seq((1, 2), (3, 4))

        ```
        If the length of seq is not evenly divisible by n, the final tuple is dropped if pad is not specified, or filled to length n by pad:
        ```python
        >>> pc.Iter([1, 2, 3, 4, 5]).partition(2).collect()
        Seq((1, 2), (3, 4), (5, None))

        ```
        """
        return self._iter(partial(cz.itertoolz.partition, n, pad=pad))

    def partition_all(self, n: int) -> Iter[tuple[T, ...]]:
        """Partition all elements of sequence into tuples of length at most n.

        The final tuple may be shorter to accommodate extra elements.

        Args:
            n (int): Maximum length of each partition.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3, 4]).partition_all(2).collect()
        Seq((1, 2), (3, 4))
        >>> pc.Iter([1, 2, 3, 4, 5]).partition_all(2).collect()
        Seq((1, 2), (3, 4), (5,))

        ```
        """
        return self._iter(partial(cz.itertoolz.partition_all, n))

    def partition_by(self, predicate: Callable[[T], bool]) -> Iter[tuple[T, ...]]:
        """Partition the `iterable` into a sequence of `tuples` according to a predicate function.

        Every time the output of `predicate` changes, a new `tuple` is started,
        and subsequent items are collected into that `tuple`.

        Args:
            predicate (Callable[[T], bool]): Function to determine partition boundaries.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        >>> import pyochain as pc
        >>> pc.Iter("I have space").partition_by(lambda c: c == " ").collect()
        Seq(('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e'))
        >>>
        >>> data = [1, 2, 1, 99, 88, 33, 99, -1, 5]
        >>> pc.Iter(data).partition_by(lambda x: x > 10).collect()
        Seq((1, 2, 1), (99, 88, 33, 99), (-1, 5))

        ```
        """
        return self._iter(partial(cz.recipes.partitionby, predicate))

    def batch(self, n: int) -> Iter[tuple[T, ...]]:
        """Batch elements into tuples of length n and return a new Iter.

        - The last batch may be shorter than n.
        - The data is consumed lazily, just enough to fill a batch.
        - The result is yielded as soon as a batch is full or when the input iterable is exhausted.

        Args:
            n (int): Number of elements in each batch.

        Returns:
            Iter[tuple[T, ...]]: An iterable of batched tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("ABCDEFG").batch(3).collect()
        Seq(('A', 'B', 'C'), ('D', 'E', 'F'), ('G',))

        ```
        """
        return self._iter(itertools.batched, n)

    def cycle(self) -> Iter[T]:
        """Repeat the sequence indefinitely.

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use Iter.take() or Iter.slice() to limit the number of items taken.

        Returns:
            Iter[T]: A new Iterable wrapper that cycles through the elements indefinitely.
        ```python

        Example:
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).cycle().take(5).collect()
        Seq(1, 2, 1, 2, 1)

        ```
        """
        return self._iter(itertools.cycle)

    def interpose(self, element: T) -> Iter[T]:
        """Interpose element between items and return a new Iterable wrapper.

        Args:
            element (T): The element to interpose between items.

        Returns:
            Iter[T]: A new Iterable wrapper with the element interposed.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).interpose(0).collect()
        Seq(1, 0, 2)

        ```
        """
        return self._iter(partial(cz.itertoolz.interpose, element))

    def random_sample(
        self,
        probability: float,
        state: Random | int | None = None,
    ) -> Iter[T]:
        """Return elements from a sequence with probability of prob.

        Returns a lazy iterator of random items from seq.

        random_sample considers each item independently and without replacement.

        See below how the first time it returned 13 items and the next time it returned 6 items.

        Args:
            probability (float): The probability of including each element.
            state (Random | int | None): Random state or seed for deterministic sampling.

        Returns:
            Iter[T]: A new Iterable wrapper with randomly sampled elements.
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
        return self._iter(
            partial(cz.itertoolz.random_sample, probability, random_state=state),
        )

    def accumulate(self, func: Callable[[T, T], T]) -> Iter[T]:
        """Return cumulative application of binary op provided by the function.

        Args:
            func (Callable[[T, T], T]): A binary function to apply cumulatively.

        Returns:
            Iter[T]: A new Iterable wrapper with accumulated results.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 3)).accumulate(lambda a, b: a + b).collect()
        Seq(1, 3, 6)

        ```
        """
        return self._iter(partial(cz.itertoolz.accumulate, func))

    def insert_left(self, value: T) -> Iter[T]:
        """Prepend value to the sequence and return a new Iterable wrapper.

        Args:
            value (T): The value to prepend.

        Returns:
            Iter[T]: A new Iterable wrapper with the value prepended.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((2, 3)).insert_left(1).collect()
        Seq(1, 2, 3)

        ```
        """
        return self._iter(partial(cz.itertoolz.cons, value))

    def peek(self, n: int, func: Callable[[Iterable[T]], Any]) -> Iter[T]:
        """Retrieve the first n items from the iterable, pass them to func, and return the original iterable.

        Allow to pass side-effect functions that process the peeked items without consuming the original Iterator.

        Args:
            n (int): Number of items to peek.
            func (Callable[[Iterable[T]], Any]): Function to process the peeked items.

        Returns:
            Iter[T]: A new Iterable wrapper with the peeked items.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).peek(2, lambda x: print(f"Peeked {len(x)} values: {x}")).collect()
        Peeked 2 values: (1, 2)
        Seq(1, 2, 3)

        ```
        """

        def _peek(data: Iterable[T]) -> Iterator[T]:
            peeked = Peeked(*cz.itertoolz.peekn(n, data))
            func(peeked.values)
            return peeked.original

        return self._iter(_peek)

    def interleave(self, *others: Iterable[T]) -> Iter[T]:
        """Interleave multiple sequences element-wise.

        Args:
            *others (Iterable[T]): Other iterables to interleave.

        Returns:
            Iter[T]: A new Iterable wrapper with interleaved elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).interleave((3, 4)).collect()
        Seq(1, 3, 2, 4)

        ```
        """

        def _interleave(data: Iterable[T]) -> Iterator[T]:
            return cz.itertoolz.interleave((data, *others))

        return self._iter(_interleave)

    def chain(self, *others: Iterable[T]) -> Iter[T]:
        """Concatenate zero or more iterables, any of which may be infinite.

        An infinite sequence will prevent the rest of the arguments from being included.

        We use chain.from_iterable rather than chain(*seqs) so that seqs can be a generator.

        Args:
            *others (Iterable[T]): Other iterables to concatenate.

        Returns:
            Iter[T]: A new Iterable wrapper with concatenated elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2)).chain((3, 4), [5]).collect()
        Seq(1, 2, 3, 4, 5)

        ```
        """

        def _chain(data: Iterable[T]) -> Iterator[T]:
            return cz.itertoolz.concat((data, *others))  # ty:ignore[invalid-return-type] # @Todo(StarredExpression)

        return self._iter(_chain)

    def elements(self) -> Iter[T]:
        """Iterator over elements repeating each as many times as its count.

        Note:
            if an element's count has been set to zero or is a negative
            number, elements() will ignore it.

        Returns:
            Iter[T]: A new Iterable wrapper with elements repeated according to their counts.

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

        def _elements(data: Iterable[T]) -> Iterator[T]:
            return Counter(data).elements()

        return self._iter(_elements)

    def rev(self) -> Iter[T]:
        """Return a new Iterable wrapper with elements in reverse order.

        The result is a new iterable over the reversed sequence.

        Note:
            This method must consume the entire iterable to perform the reversal.

        Returns:
            Iter[T]: A new Iterable wrapper with elements in reverse order.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).rev().collect()
        Seq(3, 2, 1)

        ```
        """

        def _reverse(data: Iterable[T]) -> Iterator[T]:
            return reversed(tuple(data))

        return self._iter(_reverse)

    def is_strictly_n(self, n: int) -> Iter[Result[T, ValueError]]:
        """Yield`Ok[T]` as long as the iterable has exactly *n* items.

        If it has fewer than *n* items, yield `Err[ValueError]` with the actual number of items.

        If it has more than *n* items, yield `Err[ValueError]` with the number `n + 1`.

        Note that the returned iterable must be consumed in order for the check to
        be made.

        Args:
            n (int): The exact number of items expected.

        Returns:
            Iter[Result[T, ValueError]]: A new Iterable wrapper yielding results based on the item count.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = ["a", "b", "c", "d"]
        >>> n = 4
        >>> pc.Iter(data).is_strictly_n(n).collect()
        Seq(Ok('a'), Ok('b'), Ok('c'), Ok('d'))
        >>> pc.Iter("ab").is_strictly_n(3).collect()  # doctest: +NORMALIZE_WHITESPACE
        Seq(Ok('a'), Ok('b'),
        Err(ValueError('Too few items in iterable (got 2)')))
        >>> pc.Iter("abc").is_strictly_n(2).collect()  # doctest: +NORMALIZE_WHITESPACE
        Seq(Ok('a'), Ok('b'),
        Err(ValueError('Too many items in iterable (got at least 3)')))

        ```
        You can easily combine this with `.map(lambda r: r.map_err(...))` to handle the errors as you wish.
        ```python
        >>> def _my_err(e: ValueError) -> str:
        ...     return f"custom error: {e}"
        >>>
        >>> pc.Iter([1]).is_strictly_n(0).map(lambda r: r.map_err(_my_err)).collect()
        Seq(Err('custom error: Too many items in iterable (got at least 1)'),)

        ```
        Or use `.filter_map(...)` to only keep the `Ok` values.
        ```python
        >>> pc.Iter([1, 2, 3]).is_strictly_n(2).filter_map(lambda r: r.ok()).collect()
        Seq(1, 2)

        ```
        """
        from ._result import Err, Ok

        def _strictly_n_(iterable: Iterable[T]) -> Iterator[Result[T, ValueError]]:
            it = iter(iterable)

            sent = 0
            for item in itertools.islice(it, n):
                yield Ok(item)
                sent += 1

            if sent < n:
                e = ValueError(f"Too few items in iterable (got {sent})")
                yield Err(e)

            for _ in it:
                e = ValueError(f"Too many items in iterable (got at least {n + 1})")
                yield Err(e)

        return self._iter(_strictly_n_)

    def enumerate(self) -> Iter[Enumerated[T]]:
        """Return a `Iter` of (index, value) pairs.

        Each value in the iterable is paired with its index, starting from 0.

        The `Iter` yields `Enumerated[T]` tuples where **idx** is the index and **value[T]** is the corresponding element from the iterable.

        Returns:
            Iter[Enumerated[T]]: An iterable of (index, value) pairs.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(["a", "b"]).enumerate().collect()
        Seq((0, 'a'), (1, 'b'))

        ```
        """
        return self._iter(enumerate).map(lambda x: Enumerated(*x))

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).combinations(2).collect()
        Seq((1, 2), (1, 3), (2, 3))

        ```
        """
        return self._iter(itertools.combinations, r)

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).permutations(2).collect()
        Seq((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))

        ```
        """
        return self._iter(itertools.permutations, r)

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).combinations_with_replacement(2).collect()
        Seq((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))

        ```
        """
        return self._iter(itertools.combinations_with_replacement, r)

    def pairwise(self) -> Iter[tuple[T, T]]:
        """Return an iterator over pairs of consecutive elements.

        Returns:
            Iter[tuple[T, T]]: An iterable of pairs of consecutive elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).pairwise().collect()
        Seq((1, 2), (2, 3))

        ```
        """
        return self._iter(itertools.pairwise)

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
    def map_juxt(self, *funcs: Callable[[T], object]) -> Iter[tuple[object, ...]]:
        """Apply several functions to each item.

        Returns a new Iter where each item is a tuple of the results of applying each function to the original item.

        Args:
            *funcs (Callable[[T], object]): Functions to apply to each item.

        Returns:
            Iter[tuple[object, ...]]: An iterable of tuples containing the results of each function.
        ```python
        >>> import pyochain as pc
        >>> def is_even(n: int) -> bool:
        ...     return n % 2 == 0
        >>> def is_positive(n: int) -> bool:
        ...     return n > 0
        >>>
        >>> pc.Iter([1, -2, 3]).map_juxt(is_even, is_positive).collect()
        Seq((False, True), (True, False), (False, True))

        ```
        """
        return self._iter(partial(map, cz.functoolz.juxt(*funcs)))

    def adjacent(
        self,
        predicate: Callable[[T], bool],
        distance: int = 1,
    ) -> Iter[tuple[bool, T]]:
        """Return an iterable over (bool, item) tuples.

        Args:
            predicate (Callable[[T], bool]): Function to determine if an item satisfies the condition.
            distance (int): Number of places to consider as adjacent. Defaults to 1.

        Returns:
            Iter[tuple[bool, T]]: An iterable of (bool, item) tuples.

        The output is a sequence of tuples where the item is drawn from iterable.

        The bool indicates whether that item satisfies the predicate or is adjacent to an item that does.

        For example, to find whether items are adjacent to a 3:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(range(6)).adjacent(lambda x: x == 3).collect()
        Seq((False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5))

        ```
        Set distance to change what counts as adjacent.
        For example, to find whether items are two places away from a 3:
        ```python
        >>> pc.Iter(range(6)).adjacent(lambda x: x == 3, distance=2).collect()
        Seq((False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5))

        ```

        This is useful for contextualizing the results of a search function.

        For example, a code comparison tool might want to identify lines that have changed, but also surrounding lines to give the viewer of the diff context.

        The predicate function will only be called once for each item in the iterable.

        See also groupby_transform, which can be used with this function to group ranges of items with the same bool value.

        """
        return self._iter(partial(mit.adjacent, predicate, distance=distance))

    def classify_unique(self) -> Iter[tuple[T, bool, bool]]:
        """Classify each element in terms of its uniqueness.

        For each element in the input iterable, return a 3-tuple consisting of:

        - The element itself
        - False if the element is equal to the one preceding it in the input, True otherwise (i.e. the equivalent of unique_justseen)
        - False if this element has been seen anywhere in the input before, True otherwise (i.e. the equivalent of unique_everseen)

        This function is analogous to unique_everseen and is subject to the same performance considerations.

        Returns:
            Iter[tuple[T, bool, bool]]: An iterable of (element, is_new, is_unique) tuples.

        ```python
        >>> import pyochain as pc
        >>> pc.Iter("otto").classify_unique().collect()
        ... # doctest: +NORMALIZE_WHITESPACE
        Seq(('o', True,  True),
        ('t', True,  True),
        ('t', False, False),
        ('o', True,  False))

        ```
        """
        return self._iter(mit.classify_unique)

    def with_position(self) -> Iter[tuple[Position, T]]:
        """Return an iterable over (`Position`, `T`) tuples.

        The `Position` indicates whether the item `T` is the first, middle, last, or only element in the iterable.

        Returns:
            Iter[tuple[Position, T]]: An iterable of (`Position`, item) tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(["a", "b", "c"]).with_position().collect()
        Seq(('first', 'a'), ('middle', 'b'), ('last', 'c'))
        >>> pc.Iter(["a"]).with_position().collect()
        Seq(('only', 'a'),)

        ```
        """

        def gen(data: Iterable[T]) -> Iterator[tuple[Position, T]]:
            it = iter(data)

            try:
                first = next(it)
            except StopIteration:
                return

            try:
                second = next(it)
            except StopIteration:
                yield ("only", first)
                return
            yield ("first", first)

            current: T = second
            for nxt in it:
                yield ("middle", current)
                current = nxt
            yield ("last", current)

        return self._iter(gen)

    @overload
    def group_by(self, key: None = None) -> Iter[tuple[T, Iter[T]]]: ...
    @overload
    def group_by[R](self, key: Callable[[T], R]) -> Iter[tuple[R, Iter[T]]]: ...
    @overload
    def group_by[R](
        self, key: Callable[[T], R] | None = None
    ) -> Iter[tuple[R, Iter[T]] | tuple[T, Iter[T]]]: ...
    def group_by(
        self, key: Callable[[T], Any] | None = None
    ) -> Iter[tuple[Any, Iter[T]]]:
        """Make an `Iter` that returns consecutive keys and groups from the iterable.

        Args:
            key (Callable[[T], Any] | None): Function to compute the key for grouping. Defaults to None.

        Returns:
            Iter[tuple[Any, Iter[T]]]: An `Iter` of (key, group) tuples.

        The **key** is a function computing a key value for each element.

        If not specified or is None, **key** defaults to an identity function and returns the element unchanged.

        The `Iter` needs to already be sorted on the same key function.

        This is due to the fact that it generates a new group every time the value of the **key** function changes.

        That behavior differs from SQL's `GROUP BY` which aggregates common elements regardless of their input order.

        Note:
            Each group generated is itself an `Iter` and is fully lazy.
            If you need all the groups, you must materialize them with a `.map()` call and a closure that can materialize the group (e.g `.collect()`, `.into(list)`, etc...).

        Examples:
        ```python
        >>> import pyochain as pc
        >>> # Group even and odd numbers
        >>> (
        ... pc.Iter.from_count() # create an infinite iterator of integers
        ... .take(8) # take the first 8
        ... .map(lambda x: (x % 2 == 0, x)) # map to (is_even, value)
        ... .sort(key=lambda x: x[0]) # sort by is_even
        ... .iter() # Since sort collect to a Vec, we need to convert back to Iter
        ... .group_by(lambda x: x[0]) # group by is_even
        ... .map(lambda x: (x[0], x[1].map(lambda y: y[1]).into(list))) # extract values from groups, discarding keys, and materializing them to lists
        ... .collect() # collect the result
        ... .into(dict) # convert to dict
        ... )
        {False: [1, 3, 5, 7], True: [0, 2, 4, 6]}
        >>> # group by a common key, already sorted
        >>> data = [
        ...     {"name": "Alice", "gender": "F"},
        ...     {"name": "Bob", "gender": "M"},
        ...     {"name": "Charlie", "gender": "M"},
        ...     {"name": "Dan", "gender": "M"},
        ... ]
        >>> (
        ... pc.Iter(data)
        ... .group_by(lambda x: x["gender"]) # group by the gender key
        ... .map(lambda x: (x[0], x[1].length())) # get the length of each group
        ... .collect()
        ... )
        Seq(('F', 1), ('M', 3))

        ```
        """
        # TODO: Inventory of all group_by and follow-ups in Dict and Iter, potentially do like polars: GroupBy object

        def _group_by(
            data: Iterable[T],
        ) -> Iterator[tuple[Any, Iter[T]]]:
            return ((x, Iter(y)) for x, y in itertools.groupby(data, key))

        return self._iter(_group_by)
