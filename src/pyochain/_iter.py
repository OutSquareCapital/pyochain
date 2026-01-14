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
    MutableSequence,
    MutableSet,
    Sequence,
    ValuesView,
)
from dataclasses import dataclass
from typing import Any, Literal, Never, Self, TypeIs, overload

import cytoolz as cz

from ._types import SupportsRichComparison
from .rs import NONE, Err, Ok, Option, Result, Some
from .traits import (
    Checkable,
    Pipeable,
    PyoIterator,
    PyoMutableSequence,
    PyoSequence,
    PyoSet,
)

Position = Literal["first", "middle", "last", "only"]
"""Literal type representing the position of an item in an iterable."""


def _get_repr(data: Collection[Any]) -> str:
    from pprint import pformat

    def _repr_inner(data: Collection[Any]) -> str:
        return pformat(data, sort_dicts=False)[1:-1]

    match data:
        case set() | frozenset():
            return _repr_inner(tuple(data))
        case _:
            match len(data):
                case 0:
                    return ""
                case _:
                    return _repr_inner(data)


@dataclass(slots=True)
class Unzipped[T, V](Pipeable, Checkable):
    """Represents the result of unzipping an `Iter` of pairs into two separate `Iter`.

    Attributes:
        left (Iter[T]): An `Iter` over the first elements of the pairs.
        right (Iter[V]): An `Iter` over the second elements of the pairs.

    See Also:
        `Iter.unzip()`
    """

    left: Iter[T]
    """An `Iter` over the first elements of the pairs."""
    right: Iter[V]
    """An `Iter` over the second elements of the pairs."""

    def __bool__(self) -> bool:
        return bool(self.left) and bool(self.right)


@dataclass(slots=True)
class Peekable[T](Pipeable, Checkable):
    """Represents the result of peeking into an `Iter`.

    Inerhit from `Checkable` to provide truthiness based on whether any elements were peeked.

    Attributes:
        peek (Seq[T]): A `Seq` of the peeked elements.
        values (Iter[T]): An `Iter` of values, still including the peeked elements.

    See Also:
        `Iter.peekable()`
        `Iter.cloned()`


    """

    peek: Seq[T]
    """A `Seq` of the peeked elements."""
    values: Iter[T]
    """An `Iter` of values, still including the peeked elements."""

    def __bool__(self) -> bool:
        return bool(self.peek)


@dataclass(slots=True)
class DrainIterator[T](Iterator[T]):
    """An `Iterator` that drains elements from a `Vec` within a specified range.

    This class is not supposed to be used directly. Use `Vec.drain()` instead to obtain an `Iter` wrapper around it.

    See `Vec.drain()` for details.
    """

    _vec: MutableSequence[T]
    _idx: int
    _end_idx: int

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self._idx >= self._end_idx:
            raise StopIteration
        val = self._vec.pop(self._idx)
        self._end_idx -= 1
        return val

    def __del__(self) -> None:
        pop = self._vec.pop
        while self._idx < self._end_idx:
            pop(self._idx)
            self._end_idx -= 1


class Set[T](PyoSet[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `Collection` Protocol from `collections.abc`, so it can be used as a standard immutable collection.

    The underlying data structure is a `frozenset`.

    Tip:
        - `Set(frozenset)` is a no-copy operation since Python optimizes this under the hood.
        - If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
            data (Iterable[T]): The data to initialize the Set with.
    """

    __slots__ = ("_inner",)
    __match_args__ = ("_inner",)
    _inner: frozenset[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = frozenset(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({_get_repr(self._inner)})"

    def __contains__(self, item: object) -> bool:
        return item in self._inner

    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class SetMut[T](Set[T], MutableSet[T]):
    """A mutable `set` wrapper with functional API.

    Unlike `Set` which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    Tip:
        If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
        data (Iterable[T]): The mutable set to wrap.
    """

    __slots__ = ()
    _inner: set[T]  # type: ignore[override]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = set(data)  # type: ignore[override]

    @staticmethod
    def from_ref[V](data: set[V]) -> SetMut[V]:
        """Create a `SetMut` from a reference to an existing `set`.

        This method wraps the provided `set` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `SetMut` from foreign functions that return `set` objects.

        Warning:
            Since the `SetMut` directly references the original `set`, any modifications made to the `SetMut` will also affect the original `set`, and vice versa.

        Args:
            data (set[V]): The `set` to wrap.

        Returns:
            SetMut[V]: A new `SetMut` instance wrapping the provided `set`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> original_set = {1, 2, 3}
        >>> set_obj = pc.SetMut.from_ref(original_set)
        >>> set_obj
        SetMut(1, 2, 3)
        >>> original_set.add(4)
        >>> set_obj
        SetMut(1, 2, 3, 4)


        ```
        """
        instance: SetMut[V] = SetMut.__new__(SetMut)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    def add(self, value: T) -> None:
        """Add an element to **self**.

        Args:
            value (T): The element to add.

        Example:
        ```python
        >>> import pyochain as pc
        >>> s = pc.SetMut({'a', 'b'})
        >>> s.add('c')
        >>> s.iter().sort()
        Vec('a', 'b', 'c')

        ```
        """
        self._inner.add(value)

    def discard(self, value: T) -> None:
        """Remove an element from **self** if it is a member.

        Unlike `.remove()`, the `discard()` method does not raise an exception when an element is missing from the set.

        Args:
            value (T): The element to remove.

        Example:
        ```python
        >>> import pyochain as pc
        >>> s = pc.SetMut({'a', 'b', 'c'})
        >>> s.discard('b')
        >>> s.iter().sort()
        Vec('a', 'c')

        ```
        """
        self._inner.discard(value)


class Seq[T](PyoSequence[T]):
    """Represent an in memory `Sequence`.

    Implements the `Sequence` Protocol from `collections.abc`.

    Provides a subset of `Iter` methods with eager evaluation, and is the return type of `Iter.collect()`.

    The underlying data structure is an immutable `tuple`, hence the memory efficiency is better than a `Vec`.

    Tip:
        `Seq(tuple)` is preferred over `Seq(list)` as this is a no-copy operation (Python optimizes `tuple` creation from another `tuple`).
        If you have an existing `list`, consider using `Vec.from_ref()` instead to avoid unnecessary copying.

    Args:
            data (Iterable[T]): The data to initialize the Seq with.
    """

    __slots__ = ("_inner",)
    _inner: tuple[T, ...]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = tuple(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({_get_repr(self._inner)})"

    def __len__(self) -> int:
        return len(self._inner)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    def __getitem__(self, index: int | slice[Any, Any, Any]) -> T | Sequence[T]:
        return self._inner.__getitem__(index)


class Vec[T](Seq[T], PyoMutableSequence[T]):
    """A `MutableSequence` wrapper with functional API.

    Implement `MutableSequence` Protocol from `collections.abc`.

    Unlike `Seq` which is immutable, `Vec` allows in-place modification of elements.

    Implement the `MutableSequence` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable sequence.

    Args:
        data (Iterable[T]): The `Iterable` to wrap.
    """

    __slots__ = ()
    _inner: list[T]  # type: ignore[override]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = list(data)  # type: ignore[override]

    @staticmethod
    def from_ref[V](data: list[V]) -> Vec[V]:
        """Create a `Vec` from a reference to an existing `list`.

        This method wraps the provided `list` without copying it, allowing for efficient creation of a `Vec`.

        This is the recommended way to create a `Vec` from foreign functions.

        Warning:
            Since the `Vec` directly references the original `list`, any modifications made to the `Vec` will also affect the original `list`, and vice versa.

        Args:
            data (list[V]): The `list` to wrap.

        Returns:
            Vec[V]: A new Vec instance wrapping the provided `list`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> original_list = [1, 2, 3]
        >>> vec = pc.Vec.from_ref(original_list)
        >>> vec
        Vec(1, 2, 3)
        >>> vec[0] = 10
        >>> original_list
        [10, 2, 3]

        ```
        """
        instance: Vec[V] = Vec.__new__(Vec)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        return self._inner.__setitem__(index, value)  # type: ignore[arg-type]

    def __delitem__(self, index: int | slice) -> None:
        del self._inner[index]

    def insert(self, index: int, value: T) -> None:
        """Inserts an element at position index within the vector, shifting all elements after it to the right.

        Args:
            index (int): Position where to insert the element.
            value (T): The element to insert.

        Example:
        ```python
        >>> import pyochain as pc
        >>> vec = pc.Vec(['a', 'b', 'c'])
        >>> vec.insert(1, 'd')
        >>> vec
        Vec('a', 'd', 'b', 'c')
        >>> vec.insert(4, 'e')
        >>> vec
        Vec('a', 'd', 'b', 'c', 'e')

        ```
        """
        self._inner.insert(index, value)

    @overload
    def sort[U: SupportsRichComparison[Any]](
        self: Vec[U], *, key: None = None, reverse: bool = False
    ) -> Vec[U]: ...
    @overload
    def sort(
        self, *, key: Callable[[T], SupportsRichComparison[Any]], reverse: bool = False
    ) -> Vec[T]: ...
    @overload
    def sort(
        self,
        *,
        key: None = None,
        reverse: bool = False,
    ) -> Never: ...
    def sort(
        self,
        *,
        key: Callable[[T], SupportsRichComparison[Any]] | None = None,
        reverse: bool = False,
    ) -> Vec[Any]:
        """Sort the elements of the `Vec` in place.

        Warning:
            This method modifies the `Vec` in place and returns the same instance for chaining.

        Args:
            key (Callable[[T], SupportsRichComparison[Any]] | None): Optional function to extract a comparison key from each element.
            reverse (bool): If True, sort in descending order.

        Returns:
            Vec[Any]: The sorted `Vec` instance (self).

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Vec((3, 1, 2)).sort()
        Vec(1, 2, 3)
        >>> pc.Iter((3, 1, 2)).map(str).collect(pc.Vec).sort(key=int)
        Vec('1', '2', '3')

        """
        self._inner.sort(key=key, reverse=reverse)  # type: ignore[arg-type]
        return self

    def extract_if(
        self, predicate: Callable[[T], bool], start: int = 0, end: int | None = None
    ) -> Iter[T]:
        """Creates an `Iter` which uses a *predicate* to determine if an element in the `Vec` should be removed.

        If the *predicate* returns `True`, the element is removed from the `Vec` and yielded.

        If the *predicate* returns `False`, the element remains in the `Vec` and will not be yielded.

        You can specify a range for the extraction.

        If the returned ExtractIf is not exhausted, e.g. because it is dropped without iterating or the iteration short-circuits, then the remaining elements will be retained.

        Use retain_mut with a negated predicate if you do not need the returned iterator.

        Using this method is equivalent to the following code:
        ```python
            data = pc.Vec([ ... ])
            for i in range(data.length()):
                if predicate(data[i]):
                    val = data.pop(i)
                    # your code here
        ```
        """

        def _extract_if_gen() -> Iterator[T]:
            effective_end = end if end is not None else len(self)
            i = start
            pop = self.pop
            while i < effective_end and i < len(self):
                if predicate(self[i]):
                    yield pop(i)
                    effective_end -= 1
                else:
                    i += 1

        return Iter(_extract_if_gen())

    def drain(self, start: int | None = None, end: int | None = None) -> Iter[T]:
        """Removes the subslice indicated by the given *start* and *end* from the `Vec`, returning an `Iterator` over the removed subslice.

        If the `Iterator` is dropped before being fully consumed, it drops the remaining removed elements.

        Note:
            In CPython, remaining elements are cleaned up when the `Iterator` is garbage collected via `__del__`.
            However, in interactive environments like doctests, garbage collection may not happen immediately.
            To guarantee cleanup, fully consume the `Iterator` or explicitly call `.collect()` on it.

        Args:
            start (int | None): Starting index of the subslice to drain. Defaults to `0` if `None`.
            end (int | None): Ending index of the subslice to drain. Defaults to `len(self)` if `None`.

        Returns:
            Iter[T]: An `Iterator` over the drained elements.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> v = pc.Vec([1, 2, 3])
        >>> u = v.drain(1).collect();
        >>> v
        Vec(1)
        >>> u
        Seq(2, 3)
        >>> # A full range clears the vector, like `clear()` does
        >>> _ = v.drain().collect();
        >>> v
        Vec()

        ```
        Fully consuming the `Iterator` removes all drained elements
        ```python
        >>> import pyochain as pc
        >>> v = pc.Vec([1, 2, 3])
        >>> _ = v.drain(0, 3).collect()
        >>> v
        Vec()

        """
        return Iter(
            DrainIterator(self, start if start else 0, end if end else len(self))
        )

    def concat(self, other: list[T] | Self) -> Vec[T]:
        """Concatenate another `Vec` or `list` to **self** and return a new `Vec`.

        Note:
            This is equivalent to `list_1 + list_2` for standard lists.

        Args:
            other (list[T] | Self): The other `Vec` to concatenate.

        Returns:
            Vec[T]: The new `Vec` after concatenation.

        See Also:
            `Vec.extend()` which modifies **self** in place.

        Example:
        ```python
        >>> import pyochain as pc
        >>> v1 = pc.Vec([1, 2, 3])
        >>> v2 = [4, 5, 6] # Can also concatenate a standard list
        >>> v3 = v1.concat(v2)
        >>> v3
        Vec(1, 2, 3, 4, 5, 6)
        >>> v1.clear() # Clean up the original vec
        >>> v1
        Vec()
        >>> # New vec remains unaffected
        >>> v3
        Vec(1, 2, 3, 4, 5, 6)
        """
        match other:
            case Vec():
                data = self._inner + other._inner
            case list():
                data = self._inner + other
        return Vec.from_ref(data)


class Iter[T](PyoIterator[T]):
    """A superset around Python's built-in `Iterator` Protocol, providing a rich set of functional programming tools.

    Implements the `Iterator` Protocol from `collections.abc`, so it can be used as a standard iterator.

    It also provides a `__bool__()` method to check for emptiness without consuming elements.

    - An `Iterable` is any object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
    - An `Iterator` is an object representing a stream of data; returned by calling `iter()` on an `Iterable`.
    - Once an `Iterator` is exhausted, it cannot be reused or reset.

    It's designed around lazy evaluation, allowing for efficient processing of large datasets.

    Instanciating it from any `Iterable` (like lists, sets, generators, etc.) is free and efficient (it only calls the builtin `iter()` on the input).

    Once an `Iter` is created, it can be transformed and manipulated using a variety of chainable methods.

    However, keep in mind that `Iter` instances are single-use; once exhausted, they cannot be reused or reset.

    If you need to reuse the data, consider collecting it into a collection first with `.collect()`, or clone it `.cloned()` to create an independent copy.

    You can always convert back to an `Iter` using `.iter()` for free on any pyochain collection type.

    In general, avoid intermediate references when dealing with lazy iterators, and prioritize method chaining instead.

    Args:
        data (Iterable[T]): Any object that can be iterated over.
    """

    _inner: Iterator[T]
    __slots__ = ("_inner",)

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = iter(data)

    def __next__(self) -> T:
        return next(self._inner)

    def __bool__(self) -> bool:
        """Check if the `Iterator` has at least one element (mutates **self**).

        After calling this, the `Iterator` still contains all elements.

        Returns:
            bool: True if the `Iterator` has at least one element, False otherwise.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> it = pc.Iter([1, 2, 3])
        >>> bool(it)
        True
        >>> it.collect()  # All elements still available
        Seq(1, 2, 3)

        ```
        """
        first = tuple(itertools.islice(self._inner, 1))
        self._inner = itertools.chain(first, self._inner)
        return len(first) > 0

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
            - `Iter.cloned()` which is the instance method version of this function.

        Args:
            other (Self): An `Iter` instance to copy.

        Returns:
            Self: A new `Iter` instance that is independent from the original.

        Example:
        ```python
        >>> import pyochain as pc
        >>> original = pc.Iter([1, 2, 3])
        >>> copy = pc.Iter.from_ref(original)
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
        >>> import pyochain as pc
        >>> pc.Iter.once(42).collect()
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

        This can be considered the equivalent of `.insert()` but as a constructor.

        Unlike `.once()`, this function will lazily generate the value on request.

        Args:
            func (Callable[P, R]): The single value to yield.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Iter[R]: An iterator yielding the specified value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.once(42).collect()
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
        >>> import pyochain as pc
        >>> pc.Iter.from_count(10, 2).take(3).collect()
        Seq(10, 12, 14)

        ```
        """
        return Iter(itertools.count(start, step))

    @staticmethod
    def from_fn[R](f: Callable[[], Option[R]]) -> Iter[R]:
        """Create an `Iter` from a nullary generator function.

        The callable must return:

        - `Some(value)` to yield a value
        - `NONE` to stop


        Args:
            f (Callable[[], Option[R]]): Callable that returns the next item wrapped in `Option`.

        Returns:
            Iter[R]: An iterator yielding values produced by **f**.

        Example:
        ```python
        >>> import pyochain as pc
        >>> counter = 0
        >>> def gen() -> pc.Option[int]:
        ...     global counter
        ...     counter += 1
        ...     return pc.Some(counter) if counter < 6 else pc.NONE
        >>> pc.Iter.from_fn(gen).collect()
        Seq(1, 2, 3, 4, 5)

        ```
        """

        def _from_fn() -> Iterator[R]:
            while True:
                item = f()
                if item.is_none():
                    return
                yield item.unwrap()

        return Iter(_from_fn())

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
        >>> import pyochain as pc
        >>> def next_pow10(x: int) -> pc.Option[int]:
        ...     return pc.Some(x * 10) if x < 10_000 else pc.NONE
        >>> pc.Iter.successors(pc.Some(1), next_pow10).collect()
        Seq(1, 10, 100, 1000, 10000)

        ```
        """

        def _successors() -> Iterator[U]:
            current = first
            while current.is_some():
                value = current.unwrap()
                yield value
                current = succ(value)

        return Iter(_successors())

    def collect[R: Collection[Any]](
        self, collector: Callable[[Iterator[T]], R] = Seq[T]
    ) -> R:
        """Transforms an `Iter` into a collection.

        The most basic pattern in which collect() is used is to turn one collection into another.

        You take a collection, call `iter()` on it, do a bunch of transformations, and then `collect()` at the end.

        You can specify the target collection type by providing a **collector** function or type.

        This can be any `Callable` that takes an `Iterator[T]` and returns a `Collection[T]` of those types.

        Note:
            This can be tought as `.into()` with a default value (`Seq[T]`), and a different constraint (`Collection[Any]`).
            However, the runtime behavior is identical in both cases: pass **self** to the provided function, return the result.

        Args:
            collector (Callable[[Iterator[T]], R]): Function|type that defines the target collection. `R` is constrained to a `Collection`.

        Returns:
            R: A materialized collection containing the collected elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(range(5)).collect()
        Seq(0, 1, 2, 3, 4)
        >>> iterator = pc.Iter((1, 2, 3))
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
        [0, 1, 2, 3, 4]
        >>> pc.Iter(range(5)).collect(pc.Vec)
        Vec(0, 1, 2, 3, 4)
        >>> iterator = pc.Iter([1, 2, 3])
        >>> iterator._inner.__class__.__name__
        'list_iterator'

        ```
        """
        return collector(self._inner)

    @overload
    def collect_into(self, collection: Vec[T]) -> Vec[T]: ...
    @overload
    def collect_into(self, collection: list[T]) -> list[T]: ...
    def collect_into(self, collection: MutableSequence[T]) -> MutableSequence[T]:
        """Collects all the items from the `Iterator` into a `MutableSequence`.

        This method consumes the `Iterator` and adds all its items to the passed `MutableSequence`.

        The `MutableSequence` is then returned, so the call chain can be continued.

        This is useful when you already have a `MutableSequence` and want to add the `Iterator` items to it.

        This method is a convenience method to call `MutableSequence.extend()`, but instead of being called on a `MutableSequence`, it's called on an `Iterator`.

        Args:
            collection (MutableSequence[T]): A mutable collection to collect items into.

        Returns:
            MutableSequence[T]: The same mutable collection passed as argument, now containing the collected items.

        Example:
        Basic usage:
        ```python
        >>> import pyochain as pc
        >>> a = pc.Seq([1, 2, 3])
        >>> vec = pc.Vec([0, 1])
        >>> a.iter().map(lambda x: x * 2).collect_into(vec)
        Vec(0, 1, 2, 4, 6)
        >>> a.iter().map(lambda x: x * 10).collect_into(vec)
        Vec(0, 1, 2, 4, 6, 10, 20, 30)

        ```
        The returned mutable sequence can be used to continue the call chain:
        ```python
        >>> import pyochain as pc
        >>> a = pc.Seq([1, 2, 3])
        >>> vec = pc.Vec[int].new()
        >>> a.iter().collect_into(vec).length() == vec.length()
        True
        >>> a.iter().collect_into(vec).length() == vec.length()
        True

        ```
        """
        collection.extend(self._inner)
        return collection

    def try_collect[U](self: Iter[Option[U]] | Iter[Result[U, Any]]) -> Option[Vec[U]]:
        """Fallibly transforms **self** into a `Vec`, short circuiting if a failure is encountered.

        `try_collect()` is a variation of `collect()` that allows fallible conversions during collection.

        Its main use case is simplifying conversions from iterators yielding `Option[T]` or `Result[T, E]` into `Option[Vec[T]]`.

        Also, if a failure is encountered during `try_collect()`, the `Iter` is still valid and may continue to be used, in which case it will continue iterating starting after the element that triggered the failure.

        See the last example below for an example of how this works.

        Note:
            This method return `Vec[U]` instead of being customizable, because the underlying data structure must be mutable in order to build up the collection.

        Returns:
            Option[Vec[U]]: `Some[Vec[U]]` if all elements were successfully collected, or `NONE` if a failure was encountered.

        Example:
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
        >>> def external_fn(x: int) -> pc.Option[int]:
        ...     if x % 2 == 0:
        ...         return pc.Some(x)
        ...     return pc.NONE
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
        collected: list[U] = []
        collected_add = collected.append
        for item in self._inner:
            match item:
                case Ok(val) | Some(val):
                    collected_add(val)
                case _:
                    return NONE
        return Some(Vec.from_ref(collected))

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

        def _chunks() -> Iterator[Self]:
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

            new = self.__class__
            while True:
                # Create new chunk
                chunk, materialize_next = _ichunk(self._inner, size)

                # Check to see whether we're at the end of the source iterable
                if not materialize_next(size):
                    return

                yield new(chunk)
                materialize_next(size)

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).flat_map(lambda x: range(x)).collect()
        Seq(0, 0, 1, 0, 1, 2)

        ```
        """
        return Iter(itertools.chain.from_iterable(map(func, self._inner)))

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

        pool: tuple[Iterable[U], ...] = tuple(self._inner)
        counts: Counter[U] = Counter(itertools.chain.from_iterable(map(set, pool)))
        uniques: set[U] = {element for element in counts if counts[element] == 1}

        return Iter((Iter(filter(uniques.__contains__, it))) for it in pool)

    def split_into(self, *sizes: Option[int]) -> Iter[Self]:
        """Yield a list of sequential items from iterable of length 'n' for each integer 'n' in sizes.

        Args:
            *sizes (Option[int]): `Some` integers specifying the sizes of each chunk. Use `NONE` for the remainder.

        Returns:
            Iter[Self]: An iterator of iterators, each containing a chunk of the original iterable.

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

        def _split_into(data: Iterator[T]) -> Iterator[Self]:
            """Credits: more_itertools.split_into."""
            new = self.__class__
            for size in sizes:
                if size.is_none():
                    yield new(data)
                    return
                else:
                    yield new(itertools.islice(data, size.unwrap()))

        return Iter(_split_into(self._inner))

    def split_when(
        self,
        predicate: Callable[[T, T], bool],
        max_split: int = -1,
    ) -> Iter[Self]:
        """Split iterable into pieces based on the output of a predicate function.

        By default, no limit is placed on the number of splits.

        Args:
            predicate (Callable[[T, T], bool]): Function that takes successive pairs of items and returns True if the iterable should be split.
            max_split (int): Maximum number of splits to perform.

        Returns:
            Iter[Self]: An iterator of iterators of items.

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

        def _split_when(data: Iterator[T], max_split: int) -> Iterator[Self]:
            """Credits: more_itertools.split_when."""
            new = self.__class__
            if max_split == 0:
                yield self
                return
            try:
                cur_item = next(data)
            except StopIteration:
                return

            buf = [cur_item]
            for next_item in data:
                if predicate(cur_item, next_item):
                    yield new(buf)
                    if max_split == 1:
                        yield new((next_item, *data))
                        return
                    buf = []
                    max_split -= 1

                buf.append(next_item)
                cur_item = next_item

            yield new(buf)

        return Iter(_split_when(self._inner, max_split))

    def split_at(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
        *,
        keep_separator: bool = False,
    ) -> Iter[Self]:
        """Yield iterators of items from iterable, where each iterator is delimited by an item where `predicate` returns True.

        By default, no limit is placed on the number of splits.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform.
            keep_separator (bool): Whether to include the separator in the output.

        Returns:
            Iter[Self]: An iterator of iterators, each containing a segment of the original iterable.

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

        def _split_at(data: Iterator[T], max_split: int) -> Iterator[Self]:
            """Credits: more_itertools.split_at."""
            new = self.__class__
            if max_split == 0:
                yield self
                return

            buf: list[T] = []
            for item in data:
                if predicate(item):
                    yield new(buf)
                    if keep_separator:
                        yield new((item,))
                    if max_split == 1:
                        yield new(data)
                        return
                    buf = []
                    max_split -= 1
                else:
                    buf.append(item)
            yield new(buf)

        return Iter(_split_at(self._inner, max_split))

    def split_after(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
    ) -> Iter[Self]:
        """Yield iterator of items from iterable, where each iterator ends with an item where `predicate` returns True.

        By default, no limit is placed on the number of splits.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform.

        Returns:
            Iter[Self]: An iterable of lists of items.

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

        def _split_after(data: Iterator[T], max_split: int) -> Iterator[Self]:
            """Credits: more_itertools.split_after."""
            new = self.__class__
            if max_split == 0:
                yield new(data)
                return

            buf: list[T] = []
            for item in data:
                buf.append(item)
                if predicate(item) and buf:
                    yield new(buf)
                    if max_split == 1:
                        buf = list(data)
                        if buf:
                            yield new(buf)
                        return
                    buf = []
                    max_split -= 1
            if buf:
                yield new(buf)

        return Iter(_split_after(self._inner, max_split))

    def split_before(
        self,
        predicate: Callable[[T], bool],
        max_split: int = -1,
    ) -> Iter[Self]:
        """Yield iterator of items from iterable, where each iterator ends with an item where `predicate` returns True.

        By default, no limit is placed on the number of splits.

        Args:
            predicate (Callable[[T], bool]): Function to determine the split points.
            max_split (int): Maximum number of splits to perform.

        Returns:
            Iter[Self]: An iterable of lists of items.


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

        def _split_before(data: Iterator[T], max_split: int) -> Iterator[Self]:
            """Credits: more_itertools.split_before."""
            new = self.__class__

            if max_split == 0:
                yield new(data)
                return

            buf: list[T] = []
            for item in data:
                if predicate(item) and buf:
                    yield new(buf)
                    if max_split == 1:
                        yield new([item, *data])
                        return
                    buf = []
                    max_split -= 1
                buf.append(item)
            if buf:
                yield new(buf)

        return Iter(_split_before(self._inner, max_split))

    def find_map[R](self, func: Callable[[T], Option[R]]) -> Option[R]:
        """Applies function to the elements of the `Iterator` and returns the first Some(R) result.

        `Iter.find_map(f)` is equivalent to `Iter.filter_map(f).next()`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element, returning an `Option[R]`.

        Returns:
            Option[R]: The first `Some(R)` result from applying `func`, or `NONE` if no such result is found.

        Example:
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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).map(lambda x: x + 1).collect()
        Seq(2, 3)
        >>> # You can use methods on the class rather than on instance for convenience:
        >>> pc.Iter(["a", "b", "c"]).map(str.upper).collect()
        Seq('A', 'B', 'C')
        >>> pc.Iter(["a", "b", "c"]).map(lambda s: s.upper()).collect()
        Seq('A', 'B', 'C')

        ```
        """
        return Iter(map(func, self._inner))

    @overload
    def map_star[R](
        self: Iter[tuple[Any]],
        func: Callable[[Any], R],
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
    def map_star[U: Iterable[Any], R](
        self: Iter[U],
        func: Callable[..., R],
    ) -> Iter[R]:
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
        >>> import pyochain as pc
        >>> def make_sku(color: str, size: str) -> str:
        ...     return f"{color}-{size}"
        >>> data = pc.Seq(["blue", "red"])
        >>> data.iter().product(["S", "M"]).map_star(make_sku).collect()
        Seq('blue-S', 'blue-M', 'red-S', 'red-M')
        >>> # This is equivalent to:
        >>> data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).collect()
        Seq('blue-S', 'blue-M', 'red-S', 'red-M')

        ```
        """
        return Iter(itertools.starmap(func, self._inner))

    def map_while[R](self, func: Callable[[T], Option[R]]) -> Iter[R]:
        """Creates an iterator that both yields elements based on a predicate and maps.

        `map_while()` takes a closure as an argument. It will call this closure on each element of
        the iterator, and yield elements while it returns `Some(_)`.

        After `NONE` is returned, `map_while()` stops and the rest of the elements are ignored.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element that returns `Option[R]`.

        Returns:
            Iter[R]: An iterator of transformed elements until `NONE` is encountered.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def checked_div(x: int) -> pc.Option[int]:
        ...     return pc.Some(16 // x) if x != 0 else pc.NONE
        >>>
        >>> data = pc.Iter([-1, 4, 0, 1])
        >>> data.map_while(checked_div).collect()
        Seq(-16, 4)
        >>> data = pc.Iter([0, 1, 2, -3, 4, 5, -6])
        >>> # Convert to positive ints, stop at first negative
        >>> data.map_while(lambda x: pc.Some(x) if x >= 0 else pc.NONE).collect()
        Seq(0, 1, 2)

        ```
        """

        def _gen() -> Generator[R]:
            for opt in map(func, self._inner):
                if opt.is_none():
                    return
                yield opt.unwrap()

        return Iter(_gen())

    def repeat(self, n: int | None = None) -> Iter[Self]:
        """Repeat the entire `Iter` **n** times (as elements).

        If **n** is `None`, repeat indefinitely.

        Operates lazily, hence if you need to get the underlying elements, you will need to collect each repeated `Iter` via `.map(lambda x: x.collect())` or similar.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            `Iter.cycle()` to repeat the *elements* of the `Iter` indefinitely.

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[Self]: An `Iter` of repeated `Iter`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2]).repeat(3).map(list).collect()
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
            current: U = initial
            for item in data:
                res = func(current, item)
                if res.is_none():
                    break
                current = res.unwrap()
                yield res.unwrap()

        return Iter(gen(self._inner))

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
        >>> # See the equivalence of next and find:
        >>> pc.Iter(data).filter(lambda x: x > 1).next()
        Some(2)
        >>> pc.Iter(data).find(lambda x: x > 1)
        Some(2)
        >>> # Using TypeIs to narrow type:
        >>> from typing import TypeIs
        >>> def _is_str(x: object) -> TypeIs[str]:
        ...     return isinstance(x, str)
        >>> mixed_data = [1, "two", 3.0, "four"]
        >>> pc.Iter(mixed_data).filter(_is_str).collect()
        Seq('two', 'four')

        ```
        """
        return Iter(filter(func, self._inner))

    @overload
    def filter_star(
        self: Iter[tuple[Any]],
        func: Callable[[Any], bool],
    ) -> Iter[tuple[Any]]: ...
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

    def filter_star[U: Iterable[Any]](
        self: Iter[U],
        func: Callable[..., bool],
    ) -> Iter[U]:
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
        >>> import pyochain as pc
        >>> data = pc.Seq(["apple", "banana", "cherry", "date"])
        >>> data.iter().enumerate().filter_star(lambda index, fruit: index % 2 == 0).map_star(lambda index, fruit: fruit.title()).collect()
        Seq('Apple', 'Cherry')

        ```
        """
        return Iter(filter(lambda x: func(*x), self._inner))

    @overload
    def filter_false[U](self, func: Callable[[T], TypeIs[U]]) -> Iter[U]: ...
    @overload
    def filter_false(self, func: Callable[[T], bool]) -> Iter[T]: ...
    def filter_false[U](
        self, func: Callable[[T], bool | TypeIs[U]]
    ) -> Iter[T] | Iter[U]:
        """Return elements for which **func** is `False`.

        The **func** can return a `TypeIs` to narrow the type of the returned `Iter`.

        This won't have any runtime effect, but allows for better type inference.

        Args:
            func (Callable[[T], bool | TypeIs[U]]): Function to evaluate each item.

        Returns:
            Iter[T] | Iter[U]: An `Iter` of the items that do not satisfy the predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).filter_false(lambda x: x > 1).collect()
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

        return Iter(_filter_map(self._inner))

    @overload
    def filter_map_star[R](
        self: Iter[tuple[Any]],
        func: Callable[[Any], Option[R]],
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
    def filter_map_star[U: Iterable[Any], R](
        self: Iter[U],
        func: Callable[..., Option[R]],
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
        >>> import pyochain as pc
        >>> data = pc.Seq([("1", "10"), ("two", "20"), ("3", "thirty")])
        >>> def _parse_pair(s1: str, s2: str) -> pc.Result[tuple[int, int], str]:
        ...     try:
        ...         return pc.Ok((int(s1), int(s2)))
        ...     except ValueError:
        ...         return pc.Err(f"Invalid integer pair: {s1!r}, {s2!r}")
        >>>
        >>> data.iter().filter_map_star(lambda s1, s2: _parse_pair(s1, s2).ok()).collect()
        Seq((1, 10),)

        ```
        """

        def _filter_map_star(data: Iterable[U]) -> Iterator[R]:
            for item in data:
                res = func(*item)
                if res.is_some():
                    yield res.unwrap()

        return Iter(_filter_map_star(self._inner))

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

        Note:
            `Iter.map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            *others (Iterable[Any]): Other iterables to zip with.
            strict (bool): If `True` and one of the arguments is exhausted before the others, raise a ValueError.

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
        return Iter(
            tuple(Option(t) for t in tup)
            for tup in itertools.zip_longest(self._inner, *others, fillvalue=None)
        )

    def unzip[U, V](self: Iter[tuple[U, V]]) -> Unzipped[U, V]:
        """Converts an iterator of pairs into a pair of iterators.

        Returns:
            Unzipped[U, V]: dataclass with first and second iterators.


        Returns an `Unzipped` dataclass, containing two iterators:

        - one from the left elements of the pairs
        - one from the right elements.

        This function is, in some sense, the opposite of `.zip()`.

        Note:
            Both iterators share the same underlying source.

            Values consumed by one iterator remain in the shared buffer until the other iterator consumes them too.

            This is the unavoidable cost of having two independent iterators over the same source.

        ```python
        >>> import pyochain as pc
        >>> data = [(1, "a"), (2, "b"), (3, "c")]
        >>> unzipped = pc.Iter(data).unzip()
        >>> unzipped.left.collect()
        Seq(1, 2, 3)
        >>> unzipped.right.collect()
        Seq('a', 'b', 'c')

        ```
        """
        left, right = itertools.tee(self._inner, 2)
        return Unzipped(Iter(x[0] for x in left), Iter(x[1] for x in right))

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
        >>> import pyochain as pc
        >>> it = pc.Iter([1, 2, 3])
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
        >>> pc.Iter(["blue", "red"]).product(["S", "M"]).collect()
        Seq(('blue', 'S'), ('blue', 'M'), ('red', 'S'), ('red', 'M'))
        >>> res = (
        ...     pc.Iter(["blue", "red"])
        ...     .product(["S", "M"])
        ...     .map_star(lambda color, size: f"{color}-{size}")
        ...     .collect()
        ... )
        >>> res
        Seq('blue-S', 'blue-M', 'red-S', 'red-M')
        >>> res = (
        ...     pc.Iter([1, 2, 3])
        ...     .product([10, 20])
        ...     .filter_star(lambda a, b: a * b >= 40)
        ...     .map_star(lambda a, b: a * b)
        ...     .collect()
        ... )
        >>> res
        Seq(40, 60)
        >>> res = (
        ...     pc.Iter([1])
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

    def diff_at[R](
        self, other: Iterable[T], key: Callable[[T], R] | None = None
    ) -> Iter[tuple[Option[T], Option[T]]]:
        """Yields pairs of differing elements from two iterables.

        Compares elements from the source iterable and another iterable at corresponding positions.

        If elements differ (based on equality or a provided key function), yields a tuple containing the differing elements wrapped in `Option`.

        If one iterable is shorter, yields `NONE` for missing elements.

        Args:
            other (Iterable[T]): Other `Iterable` to compare with.
            key (Callable[[T], R] | None): Function to apply to each item for comparison.

        Returns:
            Iter[tuple[Option[T], Option[T]]]: An `Iter` of item pairs containing differing elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq([1, 2, 3])
        >>> data.iter().diff_at([1, 2, 10, 100]).collect()
        Seq((Some(3), Some(10)), (NONE, Some(100)))
        >>> data.iter().diff_at([1, 2, 10, 100, 2, 6, 7]).collect() # doctest: +NORMALIZE_WHITESPACE
        Seq((Some(3), Some(10)),
        (NONE, Some(100)),
        (NONE, Some(2)),
        (NONE, Some(6)),
        (NONE, Some(7)))
        >>> pc.Iter(["apples", "bananas"]).diff_at(["Apples", "Oranges"], key=str.lower).collect(list)
        [(Some('bananas'), Some('Oranges'))]

        ```
        """
        if key is None:

            def _gen_no_key() -> Iterator[tuple[Option[T], Option[T]]]:
                for first, second in itertools.zip_longest(
                    map(Some, self), map(Some, other), fillvalue=NONE
                ):
                    if first.ne(second):
                        yield first, second

            return Iter(_gen_no_key())

        def _gen_with_key() -> Iterator[tuple[Option[T], Option[T]]]:
            for first, second in itertools.zip_longest(
                map(Some, self), map(Some, other), fillvalue=NONE
            ):
                if first.map(key).ne(second.map(key)):
                    yield first, second

        return Iter(_gen_with_key())

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
        self, length: int, func: Callable[[tuple[Any, ...]], R]
    ) -> Iter[R]:
        r"""Calls the given *func* for each contiguous window of size *length* over **self**.

        The windows during mapping overlaps.

        The provided function is called with the entire window as a single tuple argument.

        Args:
            length (int): The length of each window.
            func (Callable[[tuple[Any, ...]], R]): Function to apply to each window.

        Returns:
            Iter[R]: An iterator over the outputs of func.

        See Also:
            `.map_windows_star()` for a version that unpacks the window into separate arguments.

        Example:
        ```python
        >>> import pyochain as pc
        >>> import statistics
        >>> pc.Iter([1, 2, 3, 4]).map_windows(2, statistics.mean).collect()
        Seq(1.5, 2.5, 3.5)
        >>> pc.Iter("abcd").map_windows(3, lambda window: "".join(window).upper()).collect()
        Seq('ABC', 'BCD')
        >>> pc.Iter([10, 20, 30, 40, 50]).map_windows(4, sum).collect()
        Seq(100, 140)
        >>> from pathlib import Path
        >>> pc.Iter(["home", "src", "pyochain"]).map_windows(2, lambda p: str(Path(*p))).collect()
        Seq('home\\src', 'src\\pyochain')


        ```
        """
        return Iter(map(func, cz.itertoolz.sliding_window(length, self._inner)))

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
            `.map_windows()` for a version that passes the entire window as a single tuple argument.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("abcd").map_windows_star(2, lambda x, y: f"{x}+{y}").collect()
        Seq('a+b', 'b+c', 'c+d')
        >>> pc.Iter([1, 2, 3, 4]).map_windows_star(2, lambda x, y: x + y).collect()
        Seq(3, 5, 7)

        ```
        """
        return Iter(
            itertools.starmap(func, cz.itertoolz.sliding_window(length, self._inner))
        )

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
    def partition(self, n: int, pad: T) -> Iter[tuple[T, ...]]: ...
    def partition(self, n: int, pad: T | None = None) -> Iter[tuple[T, ...]]:
        """Partition **self** into `tuples` of length **n**.

        Args:
            n (int): Length of each partition.
            pad (T | None): Value to pad the last partition if needed.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        ```python
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
        return Iter(cz.itertoolz.partition(n, self._inner, pad=pad))

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
        return Iter(cz.itertoolz.partition_all(n, self._inner))

    def partition_by(self, predicate: Callable[[T], bool]) -> Iter[tuple[T, ...]]:
        """Partition the `Iterator` into a sequence of `tuples` according to a predicate function.

        Every time the output of `predicate` changes, a new `tuple` is started,
        and subsequent items are collected into that `tuple`.

        Args:
            predicate (Callable[[T], bool]): Function to determine partition boundaries.

        Returns:
            Iter[tuple[T, ...]]: An iterable of partitioned tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter("I have space").partition_by(lambda c: c == " ").collect()
        Seq(('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e'))
        >>>
        >>> data = [1, 2, 1, 99, 88, 33, 99, -1, 5]
        >>> pc.Iter(data).partition_by(lambda x: x > 10).collect()
        Seq((1, 2, 1), (99, 88, 33, 99), (-1, 5))

        ```
        """
        return Iter(cz.recipes.partitionby(predicate, self._inner))

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
        >>> import pyochain as pc
        >>> pc.Iter("ABCDEFG").batch(3).collect()
        Seq(('A', 'B', 'C'), ('D', 'E', 'F'), ('G',))

        ```
        """
        return Iter(itertools.batched(self._inner, n, strict=strict))

    def peekable(self, n: int) -> Peekable[T]:
        """Retrieve the next **n** elements from the `Iterator`, and return a `Seq` of the retrieved elements along with the original `Iterator`, unconsumed.

        The returned `Peekable` object contains two attributes:
        - *peek*: A `Seq` of the next **n** elements.
        - *values*: An `Iter` that includes the peeked elements followed by the remaining elements of the original `Iterator`.

        `Peekable` implement `Checkable` on the *peek* attribute.

        Args:
            n (int): Number of items to peek.

        Returns:
            Peekable[T]: A `Peekable` object containing the peeked elements and the remaining iterator.

        See Also:
            `Iter.cloned()` to create an independent copy of the iterator.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Iter([1, 2, 3]).peekable(2)
        >>> data.peek
        Seq(1, 2)
        >>> data.values.collect()
        Seq(1, 2, 3)

        ```
        """
        peeked = Seq(itertools.islice(self._inner, n))
        return Peekable(peeked, Iter(itertools.chain(peeked, self._inner)))

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

        def _strictly_n_(data: Iterator[T]) -> Iterator[Result[T, ValueError]]:
            sent = 0
            for item in itertools.islice(data, n):
                yield Ok(item)
                sent += 1

            if sent < n:
                e = ValueError(f"Too few items in iterable (got {sent})")
                yield Err(e)

            for _ in data:
                e = ValueError(f"Too many items in iterable (got at least {n + 1})")
                yield Err(e)

        return Iter(_strictly_n_(self._inner))

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
        >>> import pyochain as pc
        >>> pc.Iter(["a", "b"]).enumerate().collect()
        Seq((0, 'a'), (1, 'b'))
        >>> pc.Iter(["a", "b"]).enumerate().map_star(lambda idx, val: (idx, val.upper())).collect()
        Seq((0, 'A'), (1, 'B'))

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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).combinations(2).collect()
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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).permutations(2).collect()
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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).combinations_with_replacement(2).collect()
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
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).pairwise().collect()
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
        return Iter(map(cz.functoolz.juxt(*funcs), self._inner))

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

        def gen(data: Iterator[T]) -> Iterator[tuple[Position, T]]:
            try:
                first = next(data)
            except StopIteration:
                return

            try:
                second = next(data)
            except StopIteration:
                yield ("only", first)
                return
            yield ("first", first)

            current: T = second
            for nxt in self._inner:
                yield ("middle", current)
                current = nxt
            yield ("last", current)

        return Iter(gen(self._inner))

    @overload
    def group_by(self, key: None = None) -> Iter[tuple[T, Self]]: ...
    @overload
    def group_by[K](self, key: Callable[[T], K]) -> Iter[tuple[K, Self]]: ...
    @overload
    def group_by[K](
        self, key: Callable[[T], K] | None = None
    ) -> Iter[tuple[K, Self] | tuple[T, Self]]: ...
    def group_by(
        self, key: Callable[[T], Any] | None = None
    ) -> Iter[tuple[Any | T, Self]]:
        """Make an `Iter` that returns consecutive keys and groups from the iterable.

        Args:
            key (Callable[[T], Any] | None): Function computing a key value for each element..
        If not specified or is None, **key** defaults to an identity function and returns the element unchanged.

        Returns:
            Iter[tuple[Any | T, Self]]: An `Iter` of `(key, value)` tuples.

        The values yielded are `(K, Self)` tuples, where the first element is the group key and the second element is an `Iter` of type `T` over the group values.

        The `Iter` needs to already be sorted on the same key function.

        This is due to the fact that it generates a new `Group` every time the value of the **key** function changes.

        That behavior differs from SQL's `GROUP BY` which aggregates common elements regardless of their input order.

        Warning:
            You must materialize the second element of the tuple immediately when iterating over groups.

            Because `.group_by()` uses Python's `itertools.groupby` under the hood, each group's iterator shares internal state.

            When you advance to the next group, the previous group's iterator becomes invalid and will yield empty results.

        Example:
        ```python
        >>> import pyochain as pc
        >>> # Example 1: Group even and odd numbers
        >>> (
        ... pc.Iter.from_count() # create an infinite iterator of integers
        ... .take(8) # take the first 8
        ... .map(lambda x: (x % 2 == 0, x)) # map to (is_even, value)
        ... .sort(key=lambda x: x[0]) # sort by is_even
        ... .iter() # Since sort collect to a Vec, we need to convert back to Iter
        ... .group_by(lambda x: x[0]) # group by is_even
        ... .map_star(lambda g, vals: (g, vals.map_star(lambda _, y: y).into(list))) # extract values from groups, discarding keys, and materializing them to lists
        ... .collect() # collect the result
        ... .into(dict) # convert to dict
        ... )
        {False: [1, 3, 5, 7], True: [0, 2, 4, 6]}
        >>> # Example 2: Group by a common key, already sorted
        >>> data = [
        ...     {"name": "Alice", "gender": "F"},
        ...     {"name": "Bob", "gender": "M"},
        ...     {"name": "Charlie", "gender": "M"},
        ...     {"name": "Dan", "gender": "M"},
        ... ]
        >>> (
        ... pc.Iter(data)
        ... .group_by(lambda x: x["gender"]) # group by the gender key
        ... .map_star(lambda g, vals: (g, vals.length())) # get the length of each group
        ... .collect()
        ... )
        Seq(('F', 1), ('M', 3))
        >>> # Example 3: Incorrect usage with LATE materialization:
        >>> groups = pc.Iter(["a1", "a2", "b1"]).group_by(lambda x: x[0]).collect()
        >>> # Now iterate - TOO LATE! The group iterators are consumed
        >>> for g in groups:
        ...     print(g[1].collect())  # ❌ Empty!
        Seq()
        Seq()
        >>> # Example 4: Correct usage with intermediate materialization:
        >>> groups = (
        ...     pc.Iter(["a1", "a2", "b1"])
        ...     .group_by(lambda x: x[0])
        ...     .map_star(lambda g, vals: (g, vals.collect()))  # ✅ Materialize NOW
        ...     .collect()
        ...     .iter()
        ...     .for_each(lambda x: print(f"{x[0]}: {x[1]}"))
        ... )
        a: Seq('a1', 'a2')
        b: Seq('b1',)

        ```
        """
        new = self.__class__
        return Iter((x, new(y)) for x, y in itertools.groupby(self._inner, key))

    @overload
    def sort[U: SupportsRichComparison[Any]](
        self: Iter[U],
        *,
        key: None = None,
        reverse: bool = False,
    ) -> Vec[U]: ...
    @overload
    def sort(
        self,
        *,
        key: Callable[[T], SupportsRichComparison[Any]],
        reverse: bool = False,
    ) -> Vec[T]: ...
    @overload
    def sort(
        self,
        *,
        key: None = None,
        reverse: bool = False,
    ) -> Never: ...
    def sort(
        self,
        *,
        key: Callable[[T], SupportsRichComparison[Any]] | None = None,
        reverse: bool = False,
    ) -> Vec[Any]:
        """Sort the elements of the sequence.

        If a key function is provided, it is used to extract a comparison key from each element.

        Note:
            This method must consume the entire `Iter` to perform the sort.
            The result is a new `Vec` over the sorted sequence.

        Args:
            key (Callable[[T], SupportsRichComparison[Any]] | None): Function to extract a comparison key from each element.
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[Any]: A `Vec` with elements sorted.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([3, 1, 2]).sort()
        Vec(1, 2, 3)

        ```
        """
        return Vec.from_ref(sorted(self._inner, reverse=reverse, key=key))

    def tail(self, n: int) -> Seq[T]:
        """Return a `Seq` of the last **n** elements of the `Iterator`.

        Args:
            n (int): Number of elements to return.

        Returns:
            Seq[T]: A `Seq` containing the last **n** elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).tail(2)
        Seq(2, 3)

        ```
        """
        return Seq(cz.itertoolz.tail(n, self._inner))

    def top_n(self, n: int, key: Callable[[T], Any] | None = None) -> Seq[T]:
        """Return a tuple of the top-n items according to key.

        Args:
            n (int): Number of top elements to return.
            key (Callable[[T], Any] | None): Function to extract a comparison key from each element.

        Returns:
            Seq[T]: A new Seq containing the top-n elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 3, 2]).top_n(2)
        Seq(3, 2)

        ```
        """
        return Seq(cz.itertoolz.topk(n, self._inner, key=key))

    def most_common(self, n: int | None = None) -> Vec[tuple[T, int]]:
        """Return the **n** most common elements and their counts from the `Iterator`.

        If **n** is `None`, then all elements are returned.

        Args:
            n (int | None): Number of most common elements to return. Defaults to None (all elements).

        Returns:
            Vec[tuple[T, int]]: A `Vec` containing tuples of (element, count).

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 1, 2, 3, 3, 3]).most_common(2)
        Vec((3, 3), (1, 2))

        ```
        """
        from collections import Counter

        return Vec.from_ref(Counter(self._inner).most_common(n))
