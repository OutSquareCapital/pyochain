from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import cytoolz as cz
import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter

Position = Literal["first", "middle", "last", "only"]
"""Literal type representing the position of an item in an iterable."""


class BaseTuples[T](IterWrapper[T]):
    def enumerate(self) -> Iter[tuple[int, T]]:
        """Return a Iter of (index, value) pairs.

        Returns:
            Iter[tuple[int, T]]: An iterable of (index, value) pairs.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["a", "b"]).iter().enumerate().into(list)
        [(0, 'a'), (1, 'b')]

        ```

        """
        return self._lazy(enumerate)

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
        >>> pc.Seq([1, 2, 3]).iter().combinations(2).collect()
        Seq(((1, 2), (1, 3), (2, 3)))

        ```

        """
        return self._lazy(itertools.combinations, r)

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
        >>> pc.Seq([1, 2, 3]).iter().permutations(2).collect()
        Seq(((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)))

        ```

        """
        return self._lazy(itertools.permutations, r)

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
        >>> pc.Seq([1, 2, 3]).iter().combinations_with_replacement(2).collect()
        Seq(((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)))

        ```

        """
        return self._lazy(itertools.combinations_with_replacement, r)

    def pairwise(self) -> Iter[tuple[T, T]]:
        """Return an iterator over pairs of consecutive elements.

        Returns:
            Iter[tuple[T, T]]: An iterable of pairs of consecutive elements.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).iter().pairwise().collect()
        Seq(((1, 2), (2, 3)))

        ```

        """
        return self._lazy(itertools.pairwise)

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
        >>> pc.Seq([1, -2, 3]).iter().map_juxt(is_even, is_positive).collect()
        Seq(((False, True), (True, False), (False, True)))

        ```
        """
        return self._lazy(partial(map, cz.functoolz.juxt(*funcs)))

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
        >>> pc.Seq(range(6)).iter().adjacent(lambda x: x == 3).collect()
        Seq(((False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)))

        ```
        Set distance to change what counts as adjacent.
        For example, to find whether items are two places away from a 3:
        ```python
        >>> pc.Seq(range(6)).iter().adjacent(lambda x: x == 3, distance=2).collect()
        Seq(((False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)))

        ```

        This is useful for contextualizing the results of a search function.

        For example, a code comparison tool might want to identify lines that have changed, but also surrounding lines to give the viewer of the diff context.

        The predicate function will only be called once for each item in the iterable.

        See also groupby_transform, which can be used with this function to group ranges of items with the same bool value.

        """
        return self._lazy(partial(mit.adjacent, predicate, distance=distance))

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
        >>> pc.Seq("otto").iter().classify_unique().collect()
        ... # doctest: +NORMALIZE_WHITESPACE
        Seq((('o', True,  True),
        ('t', True,  True),
        ('t', False, False),
        ('o', True,  False)))

        ```
        """
        return self._lazy(mit.classify_unique)

    @overload
    def group_by_transform(
        self,
        keyfunc: None = None,
        valuefunc: None = None,
        reducefunc: None = None,
    ) -> Iter[tuple[T, Iterator[T]]]: ...
    @overload
    def group_by_transform[U](
        self,
        keyfunc: Callable[[T], U],
        valuefunc: None,
        reducefunc: None,
    ) -> Iter[tuple[U, Iterator[T]]]: ...
    @overload
    def group_by_transform[V](
        self,
        keyfunc: None,
        valuefunc: Callable[[T], V],
        reducefunc: None,
    ) -> Iter[tuple[T, Iterator[V]]]: ...
    @overload
    def group_by_transform[U, V](
        self,
        keyfunc: Callable[[T], U],
        valuefunc: Callable[[T], V],
        reducefunc: None,
    ) -> Iter[tuple[U, Iterator[V]]]: ...
    @overload
    def group_by_transform[W](
        self,
        keyfunc: None,
        valuefunc: None,
        reducefunc: Callable[[Iterator[T]], W],
    ) -> Iter[tuple[T, W]]: ...
    @overload
    def group_by_transform[U, W](
        self,
        keyfunc: Callable[[T], U],
        valuefunc: None,
        reducefunc: Callable[[Iterator[T]], W],
    ) -> Iter[tuple[U, W]]: ...
    @overload
    def group_by_transform[V, W](
        self,
        keyfunc: None,
        valuefunc: Callable[[T], V],
        reducefunc: Callable[[Iterator[V]], W],
    ) -> Iter[tuple[T, W]]: ...
    @overload
    def group_by_transform[U, V, W](
        self,
        keyfunc: Callable[[T], U],
        valuefunc: Callable[[T], V],
        reducefunc: Callable[[Iterator[V]], W],
    ) -> Iter[tuple[U, W]]: ...
    def group_by_transform[U, V](
        self,
        keyfunc: Callable[[T], U] | None = None,
        valuefunc: Callable[[T], V] | None = None,
        reducefunc: Any = None,
    ) -> Iter[tuple[Any, ...]]:
        """An extension of ``Iter.groupby`` that can apply transformations to the grouped data.

        Args:
            keyfunc (Callable[[T], U] | None): Function to compute the key for grouping. Defaults to None.
            valuefunc (Callable[[T], V] | None): Function to transform individual items after grouping. Defaults to None.
            reducefunc (Any): Function to transform each group of items. Defaults to None.

        Returns:
            Iter[tuple[Any, ...]]: An iterable of transformed groups.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq("aAAbBBcCC").iter()
        >>> data.group_by_transform(
        ...     lambda k: k.upper(), lambda v: v.lower(), lambda g: "".join(g)
        ... ).collect()
        Seq((('A', 'aaa'), ('B', 'bbb'), ('C', 'ccc')))

        ```
        Each optional argument defaults to an identity function if not specified.

        group_by_transform is useful when grouping elements of an iterable using a separate iterable as the key.

        To do this, zip the iterables and pass a keyfunc that extracts the first element and a valuefunc that extracts the second element:

        Note that the order of items in the iterable is significant.

        Only adjacent items are grouped together, so if you don't want any duplicate groups, you should sort the iterable by the key function.

        Example:
        ```python
        >>> from operator import itemgetter
        >>> data = pc.Seq([0, 0, 1, 1, 1, 2, 2, 2, 3])
        >>> data.iter().zip("abcdefghi").group_by_transform(itemgetter(0), itemgetter(1)).map(
        ...     lambda kv: (kv[0], "".join(kv[1]))
        ... ).collect()
        Seq(((0, 'ab'), (1, 'cde'), (2, 'fgh'), (3, 'i')))

        ```

        """

        def _group_by_transform(data: Iterable[T]) -> Iterator[tuple[Any, ...]]:
            return mit.groupby_transform(data, keyfunc, valuefunc, reducefunc)

        return self._lazy(_group_by_transform)

    def with_position(self) -> Iter[tuple[Position, T]]:
        """Return an iterable over (`Position`, `T`) tuples.

        The `Position` indicates whether the item `T` is the first, middle, last, or only element in the iterable.

        Returns:
            Iter[tuple[Position, T]]: An iterable of (`Position`, item) tuples.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["a", "b", "c"]).iter().with_position().collect()
        Seq((('first', 'a'), ('middle', 'b'), ('last', 'c')))
        >>> pc.Seq(["a"]).iter().with_position().collect()
        Seq((('only', 'a'),))

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

        return self._lazy(gen)
