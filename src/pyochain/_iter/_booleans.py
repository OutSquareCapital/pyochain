from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from pyochain._results._option import Option


class BaseBool[T](IterWrapper[T]):
    def all(self, predicate: Callable[[T], bool] = lambda x: bool(x)) -> bool:
        """Tests if every element of the iterator matches a predicate.

        `Iter.all()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if they all return true, then so does `Iter.all()`.

        If any of them return false, it returns false.

        An empty iterator returns true.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item. Defaults to checking truthiness.

        Returns:
            bool: True if all elements match the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, True]).all()
        True
        >>> pc.Seq([]).all()
        True
        >>> pc.Seq([1, 0]).all()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([2, 4, 6]).all(is_even)
        True

        ```
        """

        def _all(data: Iterable[T]) -> bool:
            return all(predicate(x) for x in data)

        return self.into(_all)

    def any(self, predicate: Callable[[T], bool] = lambda x: bool(x)) -> bool:
        """Tests if any element of the iterator matches a predicate.

        `Iter.any()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if any of them return true, then so does `Iter.any()`.

        If they all return false, it returns false.

        An empty iterator returns false.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item. Defaults to checking truthiness.

        Returns:
            bool: True if any element matches the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([0, 1]).any()
        True
        >>> pc.Seq(range(0)).any()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([1, 3, 4]).any(is_even)
        True

        ```
        """

        def _any(data: Iterable[T]) -> bool:
            return any(predicate(x) for x in data)

        return self.into(_any)

    def all_equal[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Return True if all items are equal.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all items are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 1, 1]).all_equal()
        True

        ```
        A function that accepts a single argument and returns a transformed version of each input item can be specified with key:
        ```python
        >>> pc.Seq("AaaA").all_equal(key=str.casefold)
        True
        >>> pc.Seq([1, 2, 3]).all_equal(key=lambda x: x < 10)
        True

        ```
        """
        return self.into(mit.all_equal, key=key)

    def all_unique[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Returns True if all the elements of iterable are unique.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all elements are unique, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq("ABCB").all_unique()
        False

        ```
        If a key function is specified, it will be used to make comparisons.
        ```python
        >>> pc.Seq("ABCb").all_unique()
        True
        >>> pc.Seq("ABCb").all_unique(str.lower)
        False

        ```
        The function returns as soon as the first non-unique element is encountered.

        Iterables with a mix of hashable and unhashable items can be used, but the function will be slower for unhashable items

        """
        return self.into(mit.all_unique, key=key)

    def is_sorted[U](
        self,
        key: Callable[[T], U] | None = None,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns True if the items of iterable are in sorted order.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.
            reverse (bool): Whether to check for descending order. Defaults to False.
            strict (bool): Whether to enforce strict sorting (no equal elements). Defaults to False.

        Returns:
            bool: True if items are sorted according to the criteria, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["1", "2", "3", "4", "5"]).is_sorted(key=int)
        True
        >>> pc.Seq([5, 4, 3, 1, 2]).is_sorted(reverse=True)
        False

        If strict, tests for strict sorting, that is, returns False if equal elements are found:
        ```python
        >>> pc.Seq([1, 2, 2]).is_sorted()
        True
        >>> pc.Seq([1, 2, 2]).is_sorted(strict=True)
        False

        ```

        The function returns False after encountering the first out-of-order item.

        This means it may produce results that differ from the built-in sorted function for objects with unusual comparison dynamics (like math.nan).

        If there are no out-of-order items, the iterable is exhausted.
        """
        return self.into(mit.is_sorted, key=key, reverse=reverse, strict=strict)

    def find(
        self,
        predicate: Callable[[T], bool],
    ) -> Option[T]:
        """Searches for an element of an iterator that satisfies a `predicate`.

        Takes a closure that returns true or false as `predicate`, and applies it to each element of the iterator.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Option[T]: The first element satisfying the predicate. `Some(value)` if found, `NONE` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def gt_five(x: int) -> bool:
        ...     return x > 5
        >>>
        >>> def gt_nine(x: int) -> bool:
        ...     return x > 9
        >>>
        >>> pc.Seq(range(10)).find(predicate=gt_five)
        Some(value=6)
        >>> pc.Seq(range(10)).find(predicate=gt_nine).unwrap_or("missing")
        'missing'

        ```
        """
        from .._results import Option

        def _find(data: Iterable[T]) -> Option[T]:
            return Option.from_(next(filter(predicate, data), None))

        return self.into(_find)
