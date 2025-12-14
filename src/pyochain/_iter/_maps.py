from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import cytoolz as cz

from .._core import IterWrapper
from .._results import NONE, Option, Some

if TYPE_CHECKING:
    from ._main import Iter


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
        from .._iter import Iter

        return Iter(
            item if self._predicate(item) else func_else(item) for item in self._iter
        )

    def or_skip(self) -> Iter[T]:
        """Skip items not satisfying the predicate.

        All items satisfying the predicate are retained.

        Returns:
            Iter[T]: An Iter with only items satisfying the predicate.

        """
        from .._iter import Iter

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
        from .._iter import Iter

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
        from .._iter import Iter

        return Iter(self._func(item) for item in self._iter if self._predicate(item))


class BaseMap[T](IterWrapper[T]):
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
        Seq((2, 3))

        ```
        """
        return self._lazy(partial(map, func))

    def map_star[U: Iterable[Any], R](
        self: IterWrapper[U],
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
        Seq(('blue-S', 'blue-M', 'red-S', 'red-M'))
        >>> # This is equivalent to:
        >>> data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).collect()
        Seq(('blue-S', 'blue-M', 'red-S', 'red-M'))

        ```
        """
        return self._lazy(partial(itertools.starmap, func))

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
        Seq((-3, -2, -1, 0, 10, 20, 30))
        >>> data.iter().map_if(lambda x: x % 2 == 0).then(lambda x: f"{x} is even").or_skip().collect()
        Seq(('-2 is even', '0 is even', '2 is even'))

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
        Seq(((1, 2), (1, 2)))
        >>> pc.Iter([1, 2]).repeat(3, list).collect()
        Seq(([1, 2], [1, 2], [1, 2]))

        ```
        """

        def _repeat(data: Iterable[T]) -> Iterator[Iterable[T]]:
            return itertools.repeat(factory(data), n)

        return self._lazy(_repeat)

    def repeat_last(self) -> Iter[Option[T]]:
        """After the iterable is exhausted, keep yielding its last element.

        Wrap each yielded element in an Option[T].

        **Warning** ⚠️
            This creates an infinite iterator.
            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        Returns:
            Iter[Option[T]]: An iterable that yields the last element repeatedly, or default if empty

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(range(3)).repeat_last().take(5).map(lambda x: x.unwrap()).collect()
        Seq((0, 1, 2, 2, 2))

        If the iterable is empty, yield `NONE` indefinitely:
        ```python
        >>> pc.Iter(range(0)).repeat_last().take(5).collect()
        Seq((NONE, NONE, NONE, NONE, NONE))

        ```

        """

        def _repeat_last(data: Iterable[T]) -> Iterator[Option[T]]:
            _marker = object()
            item = _marker
            for item in data:
                yield Some(item)
            final = NONE if item is _marker else Some(item)
            yield from itertools.repeat(final)

        return self._lazy(_repeat_last)

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
        >>> pc.Iter([1, 2, 3, 4, 5]).scan(0, accumulate_until_limit).collect()
        Seq((1, 3, 6, 10))

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
