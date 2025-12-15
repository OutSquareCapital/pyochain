from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Sequence,
    ValuesView,
)
from typing import TYPE_CHECKING, Any, Concatenate, Self, overload

import cytoolz as cz

from ._aggregations import BaseAgg
from ._booleans import BaseBool
from ._dicts import BaseDict
from ._eager import BaseEager
from ._filters import BaseFilter
from ._joins import BaseJoins
from ._maps import BaseMap
from ._partitions import BasePartitions
from ._process import BaseProcess
from ._rolling import BaseRolling
from ._tuples import BaseTuples

if TYPE_CHECKING:
    from .._dict import Dict
    from .._results import Option


class CommonMethods[T](BaseAgg[T], BaseEager[T], BaseDict[T], BaseBool[T]):
    pass


def _convert_data[T](data: Iterable[T] | T, *more_data: T) -> Iterable[T]:
    return data if cz.itertoolz.isiterable(data) else (data, *more_data)


class Iter[T](
    BaseFilter[T],
    BaseProcess[T],
    BaseMap[T],
    BaseRolling[T],
    BaseTuples[T],
    BasePartitions[T],
    BaseJoins[T],
    CommonMethods[T],
):
    """A superset around Python's built-in `Iterator` Protocol, providing a rich set of functional programming tools.

    - An `Iterable` is any object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
    - An `Iterator` is an object representing a stream of data; returned by calling `iter()` on an `Iterable`.
    - Once an `Iterator` is exhausted, it cannot be reused or reset.

    It's designed around lazy evaluation, allowing for efficient processing of large datasets.

    - To instantiate from an `Iterable`, simply pass it to the standard constructor.
    - To instantiate from unpacked values, use the `from_` class method. This would be equivalent to the convenience of [x,y,z] syntax for lists.

    Once an `Iter` is created, it can be transformed and manipulated using a variety of chainable methods.

    However, keep in mind that `Iter` instances are single-use; once exhausted, they cannot be reused or reset.

    If you need to reuse the data, consider collecting it into a `Seq` first with `.collect()`.

    You can always convert back to an `Iter` using `Seq.iter()` for free.

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
        from .._results import Option

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
        >>> pc.Iter.from_count(10, 2).take(3).into(list)
        [10, 12, 14]

        ```
        """
        return Iter(itertools.count(start, step))

    @staticmethod
    def from_func[U](func: Callable[[U], U], value: U) -> Iter[U]:
        """Create an infinite iterator by repeatedly applying a function on an original value.

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        Args:
            func (Callable[[U], U]): Function to apply repeatedly.
            value (U): Initial value to start the iteration.

        Returns:
            Iter[U]: An iterator generating the sequence.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_func(lambda x: x + 1, 0).take(3).into(list)
        [0, 1, 2]

        ```
        """
        return Iter(cz.itertoolz.iterate(func, value))

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
        >>> data: tuple[int, ...] = (1, 2, 3)
        >>> iterator = pc.Iter.from_(data)
        >>> iterator.inner().__class__.__name__
        'tuple_iterator'
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped.inner().__class__.__name__
        'map'
        >>> mapped.collect(tuple)
        Seq((2, 4, 6))
        >>> # iterator is now exhausted
        >>> iterator.collect()
        Seq(())
        >>> # Creating from unpacked values
        >>> pc.Iter.from_(1, 2, 3).collect(tuple)
        Seq((1, 2, 3))

        ```
        """
        return Iter(_convert_data(data, *more_data))

    @staticmethod
    def unfold[S, V](seed: S, generator: Callable[[S], Option[tuple[V, S]]]) -> Iter[V]:
        """Create an iterator by repeatedly applying a generator function to an initial state.

        The `generator` function takes the current state and must return:

        - A tuple of `Some(value, new_state)` to emit the `value` and continue with the `new_state`.
        - `NONE` to stop the generation.

        This is functionally equivalent to a state-based `while` loop.

        **Warning** ⚠️
            If the `generator` function never returns `NONE`, it creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken if necessary.

        Args:
            seed (S): Initial state for the generator.
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
        >>> pc.Iter.unfold(seed=0, generator=counter_generator).into(list)
        [0, 10, 20, 30, 40]
        >>> # Example 2: Fibonacci sequence up to 100
        >>> type FibState = tuple[int, int]
        >>> def fib_generator(state: FibState) -> pc.Option[tuple[int, FibState]]:
        ...     a, b = state
        ...     if a > 100:
        ...         return pc.NONE
        ...     return pc.Some((a, (b, a + b)))
        >>> pc.Iter.unfold(seed=(0, 1), generator=fib_generator).into(list)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        >>> # Example 3: Infinite iterator (requires take())
        >>> pc.Iter.unfold(seed=1, generator=lambda s: pc.Some((s, s * 2))).take(5).into(list)
        [1, 2, 4, 8, 16]

        ```
        """
        from ._main import Iter

        def _unfold() -> Iterator[V]:
            current_seed: S = seed
            while True:
                result: Option[tuple[V, S]] = generator(current_seed)
                if result.is_none():
                    break
                value, next_seed = result.unwrap()
                yield value
                current_seed = next_seed

        return Iter(_unfold())

    def struct[**P, R, K, V](
        self: Iter[dict[K, V]],
        func: Callable[Concatenate[Dict[K, V], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[R]:
        """Apply a function to each element after wrapping it in a `Dict`.

        This is a convenience method for the common pattern of mapping a function over an `Iterable` of dictionaries.

        Args:
            func (Callable[Concatenate[Dict[K, V], P], R]): Function to apply to each wrapped dictionary.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Iter[R]: A new `Iter` instance containing the results of applying the function.

        Example:
        ```python
        >>> from typing import Any
        >>> import pyochain as pc

        >>> data: list[dict[str, Any]] = [
        ...     {"name": "Alice", "age": 30, "city": "New York"},
        ...     {"name": "Bob", "age": 25, "city": "Los Angeles"},
        ...     {"name": "Charlie", "age": 35, "city": "New York"},
        ...     {"name": "David", "age": 40, "city": "Paris"},
        ... ]
        >>>
        >>> def to_title(d: pc.Dict[str, Any]) -> pc.Dict[str, Any]:
        ...     return d.map_keys(lambda k: k.title())
        >>>
        >>> def is_young(d: pc.Dict[str, Any]) -> bool:
        ...     return d.inner().get("Age", 0) < 30
        >>>
        >>> def set_continent(d: pc.Dict[str, Any], value: str) -> dict[str, Any]:
        ...     return d.with_key("Continent", value).inner()
        >>>
        >>> def grouped_data():
        ...     return (
        ...         pc.Iter.from_(data)
        ...         .struct(to_title)
        ...         .filter_false(is_young)
        ...         .map(lambda d: d.drop("Age").with_key("Continent", "NA"))
        ...         .map_if(
        ...             lambda d: d.inner().get("City") == "Paris",
        ...         )
        ...         .then(lambda d: set_continent(d, "Europe"))
        ...         .or_else(
        ...             lambda d: set_continent(d, "America"))
        ...         .group_by(lambda d: d.get("Continent"))
        ...         .map_values(
        ...             lambda d: pc.Iter(d)
        ...             .struct(lambda d: d.drop("Continent").inner())
        ...             .into(list)
        ...         )
        ...     )
        >>> grouped_data()  # doctest: +NORMALIZE_WHITESPACE
        {'America': [{'City': 'New York', 'Name': 'Alice'},
                    {'City': 'New York', 'Name': 'Charlie'}],
        'Europe': [{'City': 'Paris', 'Name': 'David'}]}

        ```
        """
        from .._dict import Dict

        def _struct(data: Iterable[dict[K, V]]) -> Iterator[R]:
            return (func(Dict(x), *args, **kwargs) for x in data)

        return self._lazy(_struct)

    def collect(self, factory: Callable[[Iterable[T]], Sequence[T]] = tuple) -> Seq[T]:
        """Collect the elements into a `Sequence`, using the provided factory.

        Args:
            factory (Callable[[Iterable[T]], Sequence[T]]): A callable that takes an iterable and returns a Sequence. Defaults to `list`.

        Returns:
            Seq[T]: A `Seq` containing the collected elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(range(5)).collect()
        Seq((0, 1, 2, 3, 4))

        ```
        """
        return self._eager(factory)

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

    def chunks(self, size: int) -> Iter[Iter[T]]:
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
        >>> all_chunks = pc.Iter.from_count().chunks(4)
        >>> c_1, c_2, c_3 = all_chunks.next(), all_chunks.next(), all_chunks.next()
        >>> c_2.unwrap().collect()  # c_1's elements have been cached; c_3's haven't been
        Seq((4, 5, 6, 7))
        >>> c_1.unwrap().collect()
        Seq((0, 1, 2, 3))
        >>> c_3.unwrap().collect()
        Seq((8, 9, 10, 11))
        >>> pc.Seq([1, 2, 3, 4, 5, 6]).iter().chunks(3).map(lambda c: c.collect()).collect()
        Seq((Seq((1, 2, 3)), Seq((4, 5, 6))))
        >>> pc.Seq([1, 2, 3, 4, 5, 6, 7, 8]).iter().chunks(3).map(lambda c: c.collect()).collect()
        Seq((Seq((1, 2, 3)), Seq((4, 5, 6)), Seq((7, 8))))

        ```
        """

        def _chunks(data: Iterable[T], size: int) -> Iterator[Iter[T]]:
            from collections import deque
            from contextlib import suppress

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

                # Fill previous chunk's cache
                materialize_next(size)

        return self._lazy(lambda x: _chunks(x, size))

    @overload
    def flatten[U](self: Iter[Generator[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[ValuesView[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: Iter[Iterable[U]]) -> Iter[U]: ...
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
    def flatten(self: Iter[range]) -> Iter[int]: ...
    def flatten[U: Iterable[Any]](self: Iter[U]) -> Iter[Any]:
        """Flatten one level of nesting and return a new Iterable wrapper.

        This is a shortcut for `.apply(itertools.chain.from_iterable)`.

        Returns:
            Iter[Any]: An iterable of flattened elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([[1, 2], [3]]).flatten().collect()
        Seq((1, 2, 3))

        ```
        """
        return self._lazy(itertools.chain.from_iterable)

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
        >>> import pyochain as pc
        >>> data = [[1, 2], [3, 4]]
        >>> pc.Seq(data).iter().flat_map(lambda x: x + 10).collect()
        Seq((11, 12, 13, 14))

        ```
        """

        def _flat_map(data: Iterable[U]) -> map[R]:
            return map(func, itertools.chain.from_iterable(data))

        return self._lazy(_flat_map)

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
        Seq((['A'], ['C'], ['D']))

        ```

        If there are duplicates in one input iterable that aren't in the others they will be duplicated in the output.

        Input order is preserved:
        ```python
        >>> data = ("mississippi", "missouri")
        >>> pc.Seq(data).iter().unique_to_each().map(lambda x: x.into(list)).collect()
        Seq((['p', 'p'], ['o', 'u', 'r']))

        ```

        """
        from collections import Counter

        def _unique_to_each(data: Iterable[U]) -> Iterator[Iter[U]]:
            from ._main import Iter

            pool: tuple[Iterable[U], ...] = tuple(data)
            counts: Counter[U] = Counter(itertools.chain.from_iterable(map(set, pool)))
            uniques: set[U] = {element for element in counts if counts[element] == 1}
            return ((Iter(filter(uniques.__contains__, it))) for it in pool)

        return self._lazy(_unique_to_each)

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
        ...    return x.map(lambda x: x.collect(list)).collect()
        >>>
        >>> data = [1, 2, 3, 4, 5, 6]
        >>> pc.Iter(data).split_into(pc.Some(1), pc.Some(2), pc.Some(3)).into(_get_results)
        Seq((Seq([1]), Seq([2, 3]), Seq([4, 5, 6])))
        >>> pc.Iter(data).split_into(pc.Some(2), pc.Some(3)).into(_get_results)
        Seq((Seq([1, 2]), Seq([3, 4, 5])))
        >>> pc.Iter([1, 2, 3, 4]).split_into(pc.Some(1), pc.Some(2), pc.Some(3), pc.Some(4)).into(_get_results)
        Seq((Seq([1]), Seq([2, 3]), Seq([4]), Seq([])))
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        >>> pc.Iter(data).split_into(pc.Some(2), pc.Some(3), pc.NONE).into(_get_results)
        Seq((Seq([1, 2]), Seq([3, 4, 5]), Seq([6, 7, 8, 9, 0])))

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

        return self._lazy(_split_into)

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
        >>> data.iter().split_when(lambda x, y: x > y).map(lambda x: x.collect(list)).collect()
        Seq((Seq([1, 2, 3, 3]), Seq([2, 5]), Seq([2, 4]), Seq([2])))
        >>> data.iter().split_when(lambda x, y: x > y, max_split=2).map(lambda x: x.collect(list)).collect()
        Seq((Seq([1, 2, 3, 3]), Seq([2, 5]), Seq([2, 4, 2])))

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

        return self._lazy(_split_when, max_split)

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
        Seq((['a'], ['c', 'd', 'c'], ['a']))
        >>> pc.Iter(range(10)).split_at(lambda n: n % 2 == 1).into(_to_res)
        Seq(([0], [2], [4], [6], [8], []))
        >>> pc.Iter(range(10)).split_at(lambda n: n % 2 == 1, max_split=2).into(_to_res)
        Seq(([0], [2], [4, 5, 6, 7, 8, 9]))
        >>>
        >>> def cond(x: str) -> bool:
        ...     return x == "b"
        >>>
        >>> pc.Iter("abcdcba").split_at(cond, keep_separator=True).into(_to_res)
        Seq((['a'], ['b'], ['c', 'd', 'c'], ['b'], ['a']))

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

        return self._lazy(_split_at, max_split)

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
        Seq((['o', 'n', 'e', '1'], ['t', 'w', 'o', '2']))

        >>> def cond(n: int) -> bool:
        ...     return n % 3 == 0
        >>>
        >>> pc.Iter(range(10)).split_after(cond).map(list).collect()
        Seq(([0], [1, 2, 3], [4, 5, 6], [7, 8, 9]))
        >>> pc.Iter(range(10)).split_after(cond, max_split=2).map(list).collect()
        Seq(([0], [1, 2, 3], [4, 5, 6, 7, 8, 9]))

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

        return self._lazy(_split_after, max_split)

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
        Seq((['a'], ['b', 'c', 'd', 'c'], ['b', 'a']))
        >>>
        >>> def cond(n: int) -> bool:
        ...     return n % 2 == 1
        >>>
        >>> pc.Iter(range(10)).split_before(cond).map(list).collect()
        Seq(([0], [1, 2], [3, 4], [5, 6], [7, 8], [9]))
        >>> pc.Iter(range(10)).split_before(cond, max_split=2).map(list).collect()
        Seq(([0], [1, 2], [3, 4, 5, 6, 7, 8, 9]))

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

        return self._lazy(_split_before, max_split)


class Seq[T](CommonMethods[T], Sequence[T]):
    """`Seq` represent an in memory Sequence.

    Provides a subset of `Iter` methods with eager evaluation, and is the return type of `Iter.collect()`.

    You can create a `Seq` from any `Sequence` (like a list, or tuple) using the standard constructor,
    or from unpacked values using the `from_` class method.

    Doing `Seq(...).iter()` or `Iter.from_(...)` are equivalent.

    Args:
            data (Sequence[T]): The data to initialize the Seq with.
    """

    _inner: Sequence[T]

    __slots__ = ("_inner",)

    def __init__(self, data: Sequence[T]) -> None:
        self._inner = data  # pyright: ignore[reportIncompatibleVariableOverride]

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    def __getitem__(self, index: int | slice[Any, Any, Any]) -> T | Sequence[T]:
        return self._inner.__getitem__(index)

    def __len__(self) -> int:
        return len(self._inner)

    @overload
    @staticmethod
    def from_[U](data: Iterable[U]) -> Seq[U]: ...
    @overload
    @staticmethod
    def from_[U](data: U, *more_data: U) -> Seq[U]: ...
    @staticmethod
    def from_[U](data: Iterable[U] | U, *more_data: U) -> Seq[U]:
        """Create a `Seq` from an `Iterable` or unpacked values.

        Prefer using the standard constructor, as this method involves extra checks and conversions steps.

        Args:
            data (Iterable[U] | U): Iterable to convert into a sequence, or a single value.
            *more_data (U): Unpacked items to include in the sequence, if 'data' is not an Iterable.

        Returns:
            Seq[U]: A new Seq instance containing the provided data.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq.from_(1, 2, 3)
        Seq((1, 2, 3))

        ```

        """
        converted = _convert_data(data, *more_data)
        return Seq(converted if isinstance(converted, Sequence) else tuple(converted))

    def iter(self) -> Iter[T]:
        """Get an iterator over the sequence.

        Call this to switch to lazy evaluation.

        Returns:
            Iter[T]: An `Iter` instance wrapping an iterator over the sequence.
        """
        return self._lazy(iter)

    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Iterate over the elements and apply a function to each.

        Contratry to `Iter.for_each`, this method returns the same instance for chaining.

        Args:
            func (Callable[Concatenate[T, P], Any]): Function to apply to each element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Returns:
            Self: The same instance for chaining.

        Examples:
        ```python
        ```
        """
        for v in self.inner():
            func(v, *args, **kwargs)
        return self

    def is_distinct(self) -> bool:
        """Return True if all items are distinct.

        Returns:
            bool: True if all items are distinct, False otherwise.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).is_distinct()
        True

        ```
        """
        return self.into(cz.itertoolz.isdistinct)
