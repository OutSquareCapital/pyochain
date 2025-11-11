from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, TypeIs, overload

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter


class BaseFilter[T](IterWrapper[T]):
    @overload
    def filter[U](self, func: Callable[[T], TypeIs[U]]) -> Iter[U]: ...
    @overload
    def filter(self, func: Callable[[T], bool]) -> Iter[T]: ...
    def filter[U](self, func: Callable[[T], bool | TypeIs[U]]) -> Iter[T] | Iter[U]:
        """
        Return an iterator yielding those items of iterable for which function is true.

        Args:
            func: Function to evaluate each item.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3]).filter(lambda x: x > 1).into(list)
        [2, 3]

        ```
        """

        def _filter(data: Iterable[T]) -> Iterator[T]:
            return (x for x in data if func(x))

        return self._lazy(_filter)

    def filter_contain(
        self: IterWrapper[str], text: str, format: Callable[[str], str] | None = None
    ) -> Iter[str]:
        """
        Return elements that contain the given text.

        Optionally, a format function can be provided to preprocess each element before checking for the substring.

        Args:
            text: Substring to check for.
            format: Optional function to preprocess each element before checking. Defaults to None.
        Example:
        ```python
        >>> import pyochain as pc
        >>>
        >>> data = pc.Seq(["apple", "banana", "cherry", "date"])
        >>> data.iter().filter_contain("ana").into(list)
        ['banana']
        >>> data.iter().map(str.upper).filter_contain("ana", str.lower).into(list)
        ['BANANA']

        ```
        """

        def _filter_contain(data: Iterable[str]) -> Generator[str, None, None]:
            def _(x: str) -> bool:
                formatted = format(x) if format else x
                return text in formatted

            return (x for x in data if _(x))

        return self._lazy(_filter_contain)

    def filter_attr[U](self, attr: str, dtype: type[U] = object) -> Iter[U]:
        """
        Return elements that have the given attribute.

        The provided dtype is not checked at runtime for performance considerations.

        Args:
            attr: Name of the attribute to check for.
            dtype: Expected type of the attribute. Defaults to object.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(["hello", "world", 2, 5]).filter_attr("capitalize", str).into(
        ...     list
        ... )
        ['hello', 'world']

        ```
        """

        def check(data: Iterable[Any]) -> Generator[U, None, None]:
            def _(x: Any) -> TypeIs[U]:
                return hasattr(x, attr)

            return (x for x in data if _(x))

        return self._lazy(check)

    def filter_false(self, func: Callable[[T], bool]) -> Iter[T]:
        """
        Return elements for which func is false.

        Args:
            func: Function to evaluate each item.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3]).filter_false(lambda x: x > 1).into(list)
        [1]

        ```
        """
        return self._lazy(partial(itertools.filterfalse, func))

    def filter_except(
        self, func: Callable[[T], object], *exceptions: type[BaseException]
    ) -> Iter[T]:
        """
        Yield the items from iterable for which the validator function does not raise one of the specified exceptions.

        Validator is called for each item in iterable.

        It should be a function that accepts one argument and raises an exception if that item is not valid.

        If an exception other than one given by exceptions is raised by validator, it is raised like normal.

        Args:
            func: Validator function to apply to each item.
            exceptions: Exceptions to catch and ignore.
        Example:
        ```python
        >>> import pyochain as pc
        >>> iterable = ["1", "2", "three", "4", None]
        >>> pc.Iter.from_(iterable).filter_except(int, ValueError, TypeError).into(list)
        ['1', '2', '4']

        ```
        """

        def _filter_except(data: Iterable[T]) -> Iterator[T]:
            return mit.filter_except(func, data, *exceptions)

        return self._lazy(_filter_except)

    def take_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """
        Take items while predicate holds.

        Args:
            predicate: Function to evaluate each item.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 0]).take_while(lambda x: x > 0).into(list)
        [1, 2]

        ```
        """
        return self._lazy(partial(itertools.takewhile, predicate))

    def skip_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """
        Drop items while predicate holds.

        Args:
            predicate: Function to evaluate each item.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 0]).skip_while(lambda x: x > 0).into(list)
        [0]

        ```
        """
        return self._lazy(partial(itertools.dropwhile, predicate))

    def compress(self, *selectors: bool) -> Iter[T]:
        """
        Filter elements using a boolean selector iterable.

        Args:
            selectors: Boolean values indicating which elements to keep.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("ABCDEF").compress(1, 0, 1, 0, 1, 1).into(list)
        ['A', 'C', 'E', 'F']

        ```
        """
        return self._lazy(itertools.compress, selectors)

    def unique(self, key: Callable[[T], Any] | None = None) -> Iter[T]:
        """
        Return only unique elements of the iterable.

        Args:
            key: Function to transform items before comparison. Defaults to None.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3]).unique().into(list)
        [1, 2, 3]
        >>> pc.Iter.from_([1, 2, 1, 3]).unique().into(list)
        [1, 2, 3]

        ```
        Uniqueness can be defined by key keyword
        ```python
        >>> pc.Iter.from_(["cat", "mouse", "dog", "hen"]).unique(key=len).into(list)
        ['cat', 'mouse']

        ```
        """
        return self._lazy(cz.itertoolz.unique, key=key)

    def take(self, n: int) -> Iter[T]:
        """
        Creates an iterator that yields the first n elements, or fewer if the underlying iterator ends sooner.

        `Iter.take(n)` yields elements until n elements are yielded or the end of the iterator is reached (whichever happens first).

        The returned iterator is either:

        - A prefix of length n if the original iterator contains at least n elements
        - All of the (fewer than n) elements of the original iterator if it contains fewer than n elements.

        Args:
            n: Number of elements to take.
        Example:
        ```python
        >>> import pyochain as pc
        >>> data = [1, 2, 3]
        >>> pc.Iter.from_(data).take(2).into(list)
        [1, 2]
        >>> pc.Iter.from_(data).take(5).into(list)
        [1, 2, 3]

        ```
        """

        return self._lazy(partial(cz.itertoolz.take, n))

    def skip(self, n: int) -> Iter[T]:
        """
        Drop first n elements.

        Args:
            n: Number of elements to skip.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3]).skip(1).into(list)
        [2, 3]

        ```
        """
        return self._lazy(partial(cz.itertoolz.drop, n))

    def unique_justseen(self, key: Callable[[T], Any] | None = None) -> Iter[T]:
        """
        Yields elements in order, ignoring serial duplicates.

        Args:
            key: Function to transform items before comparison. Defaults to None.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("AAAABBBCCDAABBB").unique_justseen().into(list)
        ['A', 'B', 'C', 'D', 'A', 'B']
        >>> pc.Iter.from_("ABBCcAD").unique_justseen(str.lower).into(list)
        ['A', 'B', 'C', 'A', 'D']

        ```
        """
        return self._lazy(mit.unique_justseen, key=key)

    def unique_in_window(
        self, n: int, key: Callable[[T], Any] | None = None
    ) -> Iter[T]:
        """
        Yield the items from iterable that haven't been seen recently.

        The items in iterable must be hashable.

        Args:
            n: Size of the lookback window.
            key: Function to transform items before comparison. Defaults to None.
        Example:
        ```python
        >>> import pyochain as pc
        >>> iterable = [0, 1, 0, 2, 3, 0]
        >>> n = 3
        >>> pc.Iter.from_(iterable).unique_in_window(n).into(list)
        [0, 1, 2, 3, 0]

        ```
        The key function, if provided, will be used to determine uniqueness:
        ```python
        >>> pc.Iter.from_("abAcda").unique_in_window(3, key=str.lower).into(list)
        ['a', 'b', 'c', 'd', 'a']

        ```
        """
        return self._lazy(mit.unique_in_window, n, key=key)

    def extract(self, indices: Iterable[int]) -> Iter[T]:
        """
        Yield values at the specified indices.

        - The iterable is consumed lazily and can be infinite.
        - The indices are consumed immediately and must be finite.
        - Raises IndexError if an index lies beyond the iterable.
        - Raises ValueError for negative indices.

        Args:
            indices: Iterable of indices to extract values from.
        Example:
        ```python
        >>> import pyochain as pc
        >>> text = "abcdefghijklmnopqrstuvwxyz"
        >>> pc.Iter.from_(text).extract([7, 4, 11, 11, 14]).into(list)
        ['h', 'e', 'l', 'l', 'o']

        ```
        """
        return self._lazy(mit.extract, indices)

    def every(self, index: int) -> Iter[T]:
        """
        Return every nth item starting from first.

        Args:
            index: Step size for selecting items.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([10, 20, 30, 40]).every(2).into(list)
        [10, 30]

        ```
        """
        return self._lazy(partial(cz.itertoolz.take_nth, index))

    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Iter[T]:
        """
        Return a slice of the iterable.

        Args:
            start: Starting index of the slice. Defaults to None.
            stop: Ending index of the slice. Defaults to None.
            step: Step size for the slice. Defaults to None.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3, 4, 5]).slice(1, 4).into(list)
        [2, 3, 4]
        >>> pc.Iter.from_([1, 2, 3, 4, 5]).slice(step=2).into(list)
        [1, 3, 5]

        ```
        """

        def _slice(data: Iterable[T]) -> Iterator[T]:
            return itertools.islice(data, start, stop, step)

        return self._lazy(_slice)

    def filter_subclass[U: type[Any], R](
        self: IterWrapper[U], parent: type[R], keep_parent: bool = True
    ) -> Iter[type[R]]:
        """
        Return elements that are subclasses of the given class, optionally excluding the parent class itself.

        Args:
            parent: Parent class to check against.
            keep_parent: Whether to include the parent class itself. Defaults to True.
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
        >>> data = pc.Seq([A, B, C])
        >>> data.iter().filter_subclass(A).map(name).into(list)
        ['A', 'B']
        >>> data.iter().filter_subclass(A, keep_parent=False).map(name).into(list)
        ['B']

        ```
        """

        def _filter_subclass(
            data: Iterable[type[Any]],
        ) -> Generator[type[R], None, None]:
            if keep_parent:
                return (x for x in data if issubclass(x, parent))
            else:
                return (x for x in data if issubclass(x, parent) and x is not parent)

        return self._lazy(_filter_subclass)

    def filter_type[R](self, dtype: type[R]) -> Iter[R]:
        """
        Return elements that are instances of the given type.

        Args:
            dtype: Type to check against.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, "two", 3.0, "four", 5]).filter_type(int).into(list)
        [1, 5]

        ```
        """

        def _filter_type(data: Iterable[T]) -> Generator[R, None, None]:
            return (x for x in data if isinstance(x, dtype))

        return self._lazy(_filter_type)

    def filter_map[R](self, func: Callable[[T], R]) -> Iter[R]:
        """
        Apply func to every element of iterable, yielding only those which are not None.

        Args:
            func: Function to apply to each item.
        Example:
        ```python
        >>> import pyochain as pc
        >>> def to_int(s: str) -> int | None:
        ...     return int(s) if s.isnumeric() else None
        >>> elems = ["1", "a", "2", "b", "3"]
        >>> pc.Iter.from_(elems).filter_map(to_int).into(list)
        [1, 2, 3]

        ```
        """
        return self._lazy(partial(mit.filter_map, func))
