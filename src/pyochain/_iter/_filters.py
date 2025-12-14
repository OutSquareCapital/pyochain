from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Any, TypeIs, overload

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from .._results import Option
    from ._main import Iter


class BaseFilter[T](IterWrapper[T]):
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
        Seq((2, 3))
        >>> pc.Iter(data).filter(lambda x: x > 1).next()
        Some(value=2)
        >>> pc.Iter(data).find(lambda x: x > 1)
        Some(value=2)

        ```
        """

        def _filter(data: Iterable[T]) -> Iterator[T]:
            return (x for x in data if func(x))

        return self._lazy(_filter)

    def filter_attr[U](self, attr: str, dtype: type[U] = object) -> Iter[U]:  # noqa: ARG002
        """Return elements that have the given attribute.

        The provided dtype is not checked at runtime for performance considerations.

        Args:
            attr (str): Name of the attribute to check for.
            dtype (type[U]): Expected type of the attribute. Defaults to object.

        Returns:
            Iter[U]: An iterable of the items that have the specified attribute.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter(("hello", "world", 2, 5)).filter_attr("capitalize", str).collect()
        Seq(('hello', 'world'))

        ```
        """

        def check(data: Iterable[Any]) -> Generator[U]:
            return (x for x in data if hasattr(x, attr))

        return self._lazy(check)

    def filter_false(self, func: Callable[[T], bool]) -> Iter[T]:
        """Return elements for which func is false.

        Args:
            func (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Iter[T]: An iterable of the items that do not satisfy the predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([1, 2, 3]).filter_false(lambda x: x > 1).collect(list)
        Seq([1])

        ```
        """
        return self._lazy(partial(itertools.filterfalse, func))

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
        Seq((1, 2))

        ```
        """
        return self._lazy(partial(itertools.takewhile, predicate))

    def skip_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Drop items while predicate holds.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Iter[T]: An iterable of the items after skipping those for which the predicate is true.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, 2, 0)).skip_while(lambda x: x > 0).collect(list)
        Seq([0])

        ```
        """
        return self._lazy(partial(itertools.dropwhile, predicate))

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
        Seq(('A', 'C', 'E', 'F'))

        ```
        """
        return self._lazy(itertools.compress, selectors)

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
        Seq((1, 2, 3))
        >>> pc.Iter([1, 2, 1, 3]).unique().collect()
        Seq((1, 2, 3))

        ```
        Uniqueness can be defined by key keyword
        ```python
        >>> pc.Iter(["cat", "mouse", "dog", "hen"]).unique(key=len).collect()
        Seq(('cat', 'mouse'))

        ```
        """
        return self._lazy(cz.itertoolz.unique, key=key)

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
        Seq((1, 2))
        >>> pc.Iter(data).take(5).collect()
        Seq((1, 2, 3))

        ```
        """
        return self._lazy(partial(cz.itertoolz.take, n))

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
        Seq((2, 3))

        ```
        """
        return self._lazy(partial(cz.itertoolz.drop, n))

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
        Seq(('A', 'B', 'C', 'D', 'A', 'B'))
        >>> pc.Iter.from_("ABBCcAD").unique_justseen(str.lower).collect()
        Seq(('A', 'B', 'C', 'A', 'D'))

        ```
        """
        return self._lazy(mit.unique_justseen, key=key)

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
        Seq((0, 1, 2, 3, 0))

        ```
        The key function, if provided, will be used to determine uniqueness:
        ```python
        >>> pc.Iter.from_("abAcda").unique_in_window(3, key=str.lower).collect()
        Seq(('a', 'b', 'c', 'd', 'a'))

        ```
        """
        return self._lazy(mit.unique_in_window, n, key=key)

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
        Seq(('h', 'e', 'l', 'l', 'o'))

        ```
        """
        return self._lazy(mit.extract, indices)

    def every(self, index: int) -> Iter[T]:
        """Return every nth item starting from first.

        Args:
            index (int): Step size for selecting items.

        Returns:
            Iter[T]: An iterable of every nth item.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter([10, 20, 30, 40]).every(2).collect()
        Seq((10, 30))

        ```
        """
        return self._lazy(partial(cz.itertoolz.take_nth, index))

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
        Seq((2, 3, 4))
        >>> pc.Iter(data).slice(step=2).collect()
        Seq((1, 3, 5))

        ```
        """

        def _slice(data: Iterable[T]) -> Iterator[T]:
            return itertools.islice(data, start, stop, step)

        return self._lazy(_slice)

    def filter_subclass[U: type[Any], R](
        self: IterWrapper[U],
        parent: type[R],
        *,
        keep_parent: bool = True,
    ) -> Iter[type[R]]:
        """Return elements that are subclasses of the given class, optionally excluding the parent class itself.

        Args:
            parent (type[R]): Parent class to check against.
            keep_parent (bool): Whether to include the parent class itself. Defaults to True.

        Returns:
            Iter[type[R]]: A new Iterable wrapper with the filtered subclasses.

        Example:
        ```python
        >>> import pyochain as pc
        >>> class A:
        ...     pass
        >>> class B(A):
        ...     pass
        >>> class C:
        ...     pass
        >>> def name(cls: type[Any]) -> str:
        ...     return cls.__name__
        >>>
        >>> data = pc.Seq((A, B, C))
        >>> data.iter().filter_subclass(A).map(name).collect(list)
        Seq(['A', 'B'])
        >>> data.iter().filter_subclass(A, keep_parent=False).map(name).collect(list)
        Seq(['B'])

        ```
        """

        def _filter_subclass(
            data: Iterable[type[Any]],
        ) -> Generator[type[R]]:
            if keep_parent:
                return (x for x in data if issubclass(x, parent))
            return (x for x in data if issubclass(x, parent) and x is not parent)

        return self._lazy(_filter_subclass)

    def filter_type[R](self, dtype: type[R]) -> Iter[R]:
        """Return elements that are instances of the given type.

        Args:
            dtype (type[R]): Type to check against.

        Returns:
            Iter[R]: An iterable of the items that are instances of the specified type.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1, "two", 3.0, "four", 5)).filter_type(int).collect()
        Seq((1, 5))

        ```
        """

        def _filter_type(data: Iterable[T]) -> Generator[R]:
            return (x for x in data if isinstance(x, dtype))

        return self._lazy(_filter_type)

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
        Seq((1, 5))
        >>> # Equivalent to:
        >>> (
        ...     data.iter()
        ...    .map(lambda s: _parse(s).ok())
        ...    .filter(lambda s: s.is_some())
        ...    .map(lambda s: s.unwrap())
        ...    .collect()
        ... )
        Seq((1, 5))

        ```
        """

        def _filter_map(data: Iterable[T]) -> Iterator[R]:
            for item in data:
                res = func(item)
                if res.is_some():
                    yield res.unwrap()

        return self._lazy(_filter_map)
