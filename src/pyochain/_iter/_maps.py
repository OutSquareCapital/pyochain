from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    ValuesView,
)
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, overload

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from .._results import Option
    from ._main import Iter


class BaseMap[T](IterWrapper[T]):
    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the Iterator by applying a function to each element in the iterable.

        Args:
            func (Callable[Concatenate[T, P], Any]): Function to apply to each element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Returns:
            None: This is a terminal operation with no return value.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3]).for_each(lambda x: print(x))
        1
        2
        3

        ```

        """

        def _for_each(data: Iterable[T]) -> None:
            for v in data:
                func(v, *args, **kwargs)

        return self.into(_for_each)

    def map[R](self, func: Callable[[T], R]) -> Iter[R]:
        """Map each element through func and return a Iter of results.

        Args:
            func (Callable[[T], R]): Function to apply to each element.

        Returns:
            Iter[R]: An iterable of transformed elements.
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2]).map(lambda x: x + 1).into(list)
        [2, 3]

        ```
        """
        return self._lazy(partial(map, func))

    @overload
    def flat_map[U, R](
        self: IterWrapper[Iterable[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: IterWrapper[Iterator[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: IterWrapper[Sequence[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: IterWrapper[list[U]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[U, R](
        self: IterWrapper[tuple[U, ...]],
        func: Callable[[U], R],
    ) -> Iter[R]: ...
    @overload
    def flat_map[R](self: IterWrapper[range], func: Callable[[int], R]) -> Iter[R]: ...
    def flat_map[U: Iterable[Any], R](
        self: IterWrapper[U],
        func: Callable[[Any], R],
    ) -> Iter[Any]:
        """Map each element through func and flatten the result by one level.

        Args:
            func (Callable[[Any], R]): Function to apply to each element.

        Returns:
            Iter[Any]: An iterable of flattened transformed elements.
        >>> import pyochain as pc
        >>> data = [[1, 2], [3, 4]]
        >>> pc.Iter.from_(data).flat_map(lambda x: x + 10).into(list)
        [11, 12, 13, 14]

        ```
        """

        def _flat_map(data: Iterable[U]) -> map[R]:
            return map(func, itertools.chain.from_iterable(data))

        return self._lazy(_flat_map)

    def map_star[U: Iterable[Any], R](
        self: IterWrapper[U],
        func: Callable[..., R],
    ) -> Iter[R]:
        """Applies a function to each element.where each element is an iterable.

        Unlike `.map()`, which passes each element as a single argument,
        `.starmap()` unpacks each element into positional arguments for the function.

        In short, for each `element` in the sequence, it computes `func(*element)`.

        Args:
            func (Callable[..., R]): Function to apply to unpacked elements.

        Returns:
            Iter[R]: An iterable of results from applying the function to unpacked elements.
        >>> import pyochain as pc
        >>> def make_sku(color, size):
        ...     return f"{color}-{size}"
        >>> data = pc.Seq(["blue", "red"])
        >>> data.iter().product(["S", "M"]).map_star(make_sku).into(list)
        ['blue-S', 'blue-M', 'red-S', 'red-M']

        ```
        This is equivalent to:
        ```python
        >>> data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).into(list)
        ['blue-S', 'blue-M', 'red-S', 'red-M']

        ```

        - Use map_star when the performance matters (it is faster).
        - Use map with unpacking when readability matters (the types can be inferred).

        """
        return self._lazy(partial(itertools.starmap, func))

    def map_if[R](
        self,
        predicate: Callable[[T], bool],
        func: Callable[[T], R],
        func_else: Callable[[T], R] | None = None,
    ) -> Iter[R]:
        """Evaluate each item with `func` or `func_else`, based on the `predicate` result.

        If the result is equivalent to True, transform the item with func and yield it.

        Otherwise, the presence or absence of `func_else` determines the strategy:
            - If `func_else` is provided, transform the item with it and yield it.
            - If `func_else` is not provided, skip the item (do not yield it).


        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.
            func (Callable[[T], R]): Function to apply if predicate is True.
            func_else (Callable[[T], R] | None): Function to apply if predicate is False.

        Returns:
            Iter[R]: An iterable of transformed and filtered items.

        ```python
        >>> import pyochain as pc
        >>> from math import sqrt
        >>> data = pc.Seq(range(-5, 5))
        >>> data.iter().map_if(lambda x: x > 3, lambda x: f"{x} is toobig").into(list)
        ['4 is toobig']
        >>> data.iter().map_if(
        ...     lambda x: x >= 0,
        ...     lambda x: f"{sqrt(x):.2f}",
        ...     lambda x: None,
        ... ).collect()
        Seq([None, None, None, None, None, '0.00', '1.00', '1.41', '1.73', '2.00'])

        ```

        """

        def _map_if(data: Iterable[T]) -> Generator[R, None, None]:
            match func_else:
                case None:
                    return (func(item) for item in data if predicate(item))
                case _:
                    return (
                        func(item) if predicate(item) else func_else(item)
                        for item in data
                    )

        return self._lazy(_map_if)

    def map_except[R](
        self,
        func: Callable[[T], R],
        *exceptions: type[BaseException],
    ) -> Iter[R]:
        """Transform each item from iterable with function and yield the result, unless function raises one of the specified exceptions.

        Args:
            func (Callable[[T], R]): Function to apply to each item.
            *exceptions (type[BaseException]): Exceptions to catch and ignore.

        Returns:
            Iter[R]: An iterable of transformed items that don't raise specified exceptions.

        The function is called to transform each item in iterable

        If an exception other than one given by exceptions is raised by function, it is raised like normal.
        ```python
        >>> import pyochain as pc
        >>> iterable = ["1", "2", "three", "4", None]
        >>> pc.Iter.from_(iterable).map_except(int, ValueError, TypeError).into(list)
        [1, 2, 4]

        ```

        """

        def _map_except(data: Iterable[T]) -> Iterator[R]:
            return mit.map_except(func, data, *exceptions)

        return self._lazy(_map_except)

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
        >>> pc.Iter.from_([1, 2]).repeat(2).collect()
        Seq([(1, 2), (1, 2)])
        >>> pc.Iter.from_([1, 2]).repeat(3, list).collect()
        Seq([[1, 2], [1, 2], [1, 2]])

        ```
        """

        def _repeat(data: Iterable[T]) -> Iterator[Iterable[T]]:
            return itertools.repeat(factory(data), n)

        return self._lazy(_repeat)

    @overload
    def repeat_last(self, default: T) -> Iter[T]: ...
    @overload
    def repeat_last[U](self, default: U) -> Iter[T | U]: ...
    def repeat_last[U](self, default: U = None) -> Iter[T | U]:
        """After the iterable is exhausted, keep yielding its last element.

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        Args:
            default (U): Value to yield if the iterable is empty.

        Returns:
            Iter[T | U]: An iterable that yields the last element repeatedly, or default if empty
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(range(3)).repeat_last().take(5).into(list)
        [0, 1, 2, 2, 2]

        If the iterable is empty, yield default forever:
        ```python
        >>> pc.Iter.from_(range(0)).repeat_last(42).take(5).into(list)
        [42, 42, 42, 42, 42]

        ```

        """
        return self._lazy(mit.repeat_last, default)

    def ichunked(self, n: int) -> Iter[Iterator[T]]:
        """Break *iterable* into sub-iterables with *n* elements each.

        Args:
            n (int): Number of elements in each chunk.

        Returns:
            Iter[Iterator[T]]: An iterable of iterators, each yielding n elements.
        If the sub-iterables are read in order, the elements of *iterable*
        won't be stored in memory.

        If they are read out of order, :func:`itertools.tee` is used to cache
        elements as necessary.
        ```python
        >>> import pyochain as pc
        >>> all_chunks = pc.Iter.from_count().ichunked(4)
        >>> c_1, c_2, c_3 = all_chunks.next(), all_chunks.next(), all_chunks.next()
        >>> list(c_2)  # c_1's elements have been cached; c_3's haven't been
        [4, 5, 6, 7]
        >>> list(c_1)
        [0, 1, 2, 3]
        >>> list(c_3)
        [8, 9, 10, 11]

        ```

        """
        return self._lazy(mit.ichunked, n)

    @overload
    def flatten[U](self: IterWrapper[Generator[U, None, None]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[ValuesView[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[Iterable[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[Iterator[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[Sequence[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[list[U]]) -> Iter[U]: ...
    @overload
    def flatten[U](self: IterWrapper[tuple[U, ...]]) -> Iter[U]: ...
    @overload
    def flatten(self: IterWrapper[range]) -> Iter[int]: ...
    def flatten[U: Iterable[Any]](self: IterWrapper[U]) -> Iter[Any]:
        """Flatten one level of nesting and return a new Iterable wrapper.

        This is a shortcut for `.apply(itertools.chain.from_iterable)`.

        Returns:
            Iter[Any]: An iterable of flattened elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([[1, 2], [3]]).flatten().into(list)
        [1, 2, 3]

        ```
        """
        return self._lazy(itertools.chain.from_iterable)

    def pluck[U: Mapping[Any, Any]](
        self: IterWrapper[U],
        *keys: str | int,
    ) -> Iter[Any]:
        """Get an element from each item in a sequence using a nested key path.

        Args:
            *keys (str | int): Nested keys to extract values.

        Returns:
            Iter[Any]: An iterable of extracted values.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq(
        ...     [
        ...         {"id": 1, "info": {"name": "Alice", "age": 30}},
        ...         {"id": 2, "info": {"name": "Bob", "age": 25}},
        ...     ]
        ... )
        >>> data.iter().pluck("info").into(list)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        >>> data.iter().pluck("info", "name").into(list)
        ['Alice', 'Bob']

        ```
        Example: get the maximum age along with the corresponding id)
        ```python
        >>> data.iter().pluck("info", "age").zip(
        ...     data.iter().pluck("id").into(list)
        ... ).max()
        (30, 1)

        ```

        """
        getter = partial(cz.dicttoolz.get_in, keys)
        return self._lazy(partial(map, getter))

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
        >>> pc.Seq([1, 2, 3, 4, 5]).iter().scan(0, accumulate_until_limit).collect()
        Seq([1, 3, 6, 10])

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

        return self._lazy(gen)
