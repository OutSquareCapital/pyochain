from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from typing import TYPE_CHECKING, Any, Concatenate, Self, TypeIs, overload, override

import cytoolz as cz

from ._aggregations import BaseAgg
from ._booleans import BaseBool
from ._dicts import BaseDict
from ._eager import BaseEager
from ._filters import BaseFilter
from ._joins import BaseJoins
from ._lists import BaseList
from ._maps import BaseMap
from ._partitions import BasePartitions
from ._process import BaseProcess
from ._rolling import BaseRolling
from ._tuples import BaseTuples

if TYPE_CHECKING:
    from .._dict import Dict


class CommonMethods[T](BaseAgg[T], BaseEager[T], BaseDict[T]):
    pass


def _convert_data[T](data: Iterable[T] | T, *more_data: T) -> Iterable[T]:
    return data if cz.itertoolz.isiterable(data) else (data, *more_data)


def _is_sequence[T](data: Iterable[T]) -> TypeIs[Sequence[T]]:
    return hasattr(data, "__getitem__") and hasattr(data, "__len__")


class Iter[T](
    BaseBool[T],
    BaseFilter[T],
    BaseProcess[T],
    BaseMap[T],
    BaseRolling[T],
    BaseList[T],
    BaseTuples[T],
    BasePartitions[T],
    BaseJoins[T],
    CommonMethods[T],
):
    """A wrapper around Python's built-in `Iterators`/`Generators` types, providing a rich set of functional programming tools.

    - An `Iterable` is any object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
    - An `Iterator` is an object representing a stream of data; returned by calling `iter()` on an `Iterable`.
    - Once an `Iterator` is exhausted, it cannot be reused or reset.

    It's designed around lazy evaluation, allowing for efficient processing of large datasets.

    - To instantiate from a lazy `Iterator`/`Generator`, simply pass it to the standard constructor.
    - To instantiate from any `Iterable` (like a list or set), or unpacked values, use the `from_` class method.

    Once an `Iter` is created, it can be transformed and manipulated using a variety of chainable methods.

    However, keep in mind that `Iter` instances are single-use; once exhausted, they cannot be reused or reset.

    If you need to reuse the data, consider collecting it into a `Seq` first with `.collect()`.

    You can always convert back to an `Iter` using `Seq.iter()` for free.

    In general, avoid intermediate references when dealing with lazy iterators, and prioritize method chaining instead.

    Args:
        data (Iterator[T] | Generator[T, Any, Any]): An iterator or generator to wrap.
    """

    __slots__ = ("_inner",)

    def __init__(self, data: Iterator[T] | Generator[T, Any, Any]) -> None:
        self._inner = data

    def next(self) -> T:
        """Call next builtin on the underlying iterator to get the next element.

        Returns:
            T: The next element in the iterator.

        Example:
        ```python
        >>> import pyochain as pc
        >>> it = pc.Iter.from_([1, 2, 3])
        >>> it.next()
        1
        >>> it.next()
        2

        ```
        """
        return next(self.inner())

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
        Seq([])
        >>> # Creating from unpacked values
        >>> pc.Iter.from_(1, 2, 3).collect(tuple)
        Seq((1, 2, 3))

        ```
        """
        return Iter(iter(_convert_data(data, *more_data)))

    @staticmethod
    def unfold[S, V](seed: S, generator: Callable[[S], tuple[V, S] | None]) -> Iter[V]:
        """Create an iterator by repeatedly applying a generator function to an initial state.

        The `generator` function takes the current state and must return:

        - A tuple `(value, new_state)` to emit the `value` and continue with the `new_state`.
        - `None` to stop the generation.

        This is functionally equivalent to a state-based `while` loop.

        **Warning** ⚠️
            If the `generator` function never returns `None`, it creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken if necessary.

        Args:
            seed (S): Initial state for the generator.
            generator (Callable[[S], tuple[V, S] | None]): Function that generates the next value and state.

        Returns:
            Iter[V]: An iterator generating values produced by the generator function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> # Example 1: Simple counter up to 5
        >>> def counter_generator(state: int) -> tuple[int, int] | None:
        ...     if state < 5:
        ...         return (state * 10, state + 1)
        ...     return None
        >>> pc.Iter.unfold(seed=0, generator=counter_generator).into(list)
        [0, 10, 20, 30, 40]
        >>> # Example 2: Fibonacci sequence up to 100
        >>> type FibState = tuple[int, int]
        >>> def fib_generator(state: FibState) -> tuple[int, FibState] | None:
        ...     a, b = state
        ...     if a > 100:
        ...         return None
        ...     return (a, (b, a + b))
        >>> pc.Iter.unfold(seed=(0, 1), generator=fib_generator).into(list)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        >>> # Example 3: Infinite iterator (requires take())
        >>> pc.Iter.unfold(seed=1, generator=lambda s: (s, s * 2)).take(5).into(list)
        [1, 2, 4, 8, 16]

        ```
        """
        from ._main import Iter

        def _unfold() -> Iterator[V]:
            current_seed: S = seed
            while True:
                result: tuple[V, S] | None = generator(current_seed)
                if result is None:
                    break
                value, next_seed = result
                yield value
                current_seed = next_seed

        return Iter(_unfold())

    def itr[**P, R, U: Iterable[Any]](
        self: Iter[U],
        func: Callable[Concatenate[Iter[U], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[R]:
        """Apply a function to each element after wrapping it in an Iter.

        This is a convenience method for the common pattern of mapping a function over an iterable of iterables.

        Args:
            func (Callable[Concatenate[Iter[U], P], R]): Function to apply to each wrapped element.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Iter[R]: A new `Iter` instance containing the results of applying the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = [
        ...     [1, 2, 3],
        ...     [4, 5],
        ...     [6, 7, 8, 9],
        ... ]
        >>> pc.Iter.from_(data).itr(
        ...     lambda x: x.repeat(2).flatten().reduce(lambda a, b: a + b)
        ... ).into(list)
        [12, 18, 60]

        ```
        """

        def _itr(data: Iterable[U]) -> Generator[R, None, None]:
            return (func(Iter(iter(x)), *args, **kwargs) for x in data)

        return self._lazy(_itr)

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
        ...             lambda d: set_continent(d, "Europe"),
        ...             lambda d: set_continent(d, "America"),
        ...         )
        ...         .group_by(lambda d: d.get("Continent"))
        ...         .map_values(
        ...             lambda d: pc.Iter.from_(d)
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

        def _struct(data: Iterable[dict[K, V]]) -> Generator[R, None, None]:
            return (func(Dict(x), *args, **kwargs) for x in data)

        return self._lazy(_struct)

    def apply[**P, R](
        self,
        func: Callable[Concatenate[Iterable[T], P], Iterator[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[R]:
        """Apply a function to the underlying Iterator and return a new Iter instance.

        Allow to pass user defined functions that transform the iterable while retaining the Iter wrapper.

        Args:
            func (Callable[Concatenate[Iterable[T], P], Iterator[R]]): Function to apply to the underlying iterable.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Iter[R]: A new `Iter` instance containing the result of the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def double(data: Iterable[int]) -> Iterator[int]:
        ...     return (x * 2 for x in data)
        >>> pc.Iter.from_([1, 2, 3]).apply(double).into(list)
        [2, 4, 6]

        ```
        """
        return self._lazy(func, *args, **kwargs)

    def collect(self, factory: Callable[[Iterable[T]], Sequence[T]] = list) -> Seq[T]:
        """Collect the elements into a `Sequence`, using the provided factory.

        Args:
            factory (Callable[[Iterable[T]], Sequence[T]]): A callable that takes an iterable and returns a Sequence. Defaults to `list`.

        Returns:
            Seq[T]: A `Seq` containing the collected elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(range(5)).collect()
        Seq([0, 1, 2, 3, 4])

        ```
        """
        return self._eager(factory)

    @override
    def inner(self) -> Iterator[T]:
        return self._inner  # type: ignore[return-value]

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


class Seq[T](CommonMethods[T]):
    """`Seq` represent an in memory Sequence.

    Provides a subset of `Iter` methods with eager evaluation, and is the return type of `Iter.collect()`.

    You can create a `Seq` from any `Sequence` (like a list or a tuple) using the standard constructor,
    or from unpacked values using the `from_` class method.

    Doing `Seq(...).iter()` or `Iter.from_(...)` are equivalent.

    Args:
            data (Sequence[T]): The data to initialize the Seq with.
    """

    __slots__ = ("_inner",)

    def __init__(self, data: Sequence[T]) -> None:
        self._inner = data

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
        return Seq(converted if _is_sequence(converted) else tuple(converted))

    def iter(self) -> Iter[T]:
        """Get an iterator over the sequence.

        Call this to switch to lazy evaluation.

        Returns:
            Iter[T]: An `Iter` instance wrapping an iterator over the sequence.
        """
        return self._lazy(iter)

    def apply[**P, R](
        self,
        func: Callable[Concatenate[Iterable[T], P], Sequence[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Seq[R]:
        """Apply a function to the underlying `Sequence` and return a `Seq` instance.

        Allow to pass user defined functions that transform the `Sequence` while retaining the `Seq` wrapper.

        Args:
            func (Callable[Concatenate[Iterable[T], P], Sequence[R]]): Function to apply to the underlying `Sequence`.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Seq[R]: A new `Seq` instance containing the result of the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def double(data: Iterable[int]) -> Sequence[int]:
        ...     return [x * 2 for x in data]
        >>> pc.Seq([1, 2, 3]).apply(double).into(list)
        [2, 4, 6]

        ```
        """
        return self._eager(func, *args, **kwargs)

    @override
    def inner(self) -> Sequence[T]:
        return self._inner  # type: ignore[return-value]

    @override
    def into[**P, R](
        self,
        func: Callable[[Sequence[T]], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return super().into(func, *args, **kwargs)  # type: ignore[return-value]

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
