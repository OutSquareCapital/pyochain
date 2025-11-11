from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Iterable
from typing import TYPE_CHECKING, Any

import more_itertools as mit

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter


class BaseList[T](IterWrapper[T]):
    def implode(self) -> Iter[list[T]]:
        """
        Wrap each element in the iterable into a list.

        Syntactic sugar for `Iter.map(lambda x: [x])`.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(range(5)).implode().into(list)
        [[0], [1], [2], [3], [4]]

        ```
        """

        def _implode(data: Iterable[T]) -> Generator[list[T], None, None]:
            return ([x] for x in data)

        return self._lazy(_implode)

    def split_at(
        self,
        pred: Callable[[T], bool],
        maxsplit: int = -1,
        keep_separator: bool = False,
    ) -> Iter[list[T]]:
        """
        Yield lists of items from iterable, where each list is delimited by an item where callable pred returns True.

        Args:
            pred: Function to determine the split points.
            maxsplit: Maximum number of splits to perform. Defaults to -1 (no limit).
            keep_separator: Whether to include the separator in the output. Defaults to False.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("abcdcba").split_at(lambda x: x == "b").into(list)
        [['a'], ['c', 'd', 'c'], ['a']]
        >>> pc.Iter.from_(range(10)).split_at(lambda n: n % 2 == 1).into(list)
        [[0], [2], [4], [6], [8], []]

        At most *maxsplit* splits are done.

        If *maxsplit* is not specified or -1, then there is no limit on the number of splits:
        ```python
        >>> pc.Iter.from_(range(10)).split_at(lambda n: n % 2 == 1, maxsplit=2).into(
        ...     list
        ... )
        [[0], [2], [4, 5, 6, 7, 8, 9]]

        ```
        By default, the delimiting items are not included in the output.

        To include them, set *keep_separator* to `True`.
        ```python
        >>> def cond(x: str) -> bool:
        ...     return x == "b"
        >>> pc.Iter.from_("abcdcba").split_at(cond, keep_separator=True).into(list)
        [['a'], ['b'], ['c', 'd', 'c'], ['b'], ['a']]

        ```
        """
        return self._lazy(mit.split_at, pred, maxsplit, keep_separator)

    def split_after(
        self, predicate: Callable[[T], bool], max_split: int = -1
    ) -> Iter[list[T]]:
        """
        Yield lists of items from iterable, where each list ends with an item where callable pred returns True.

        Args:
            predicate: Function to determine the split points.
            max_split: Maximum number of splits to perform. Defaults to -1 (no limit).
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("one1two2").split_after(str.isdigit).into(list)
        [['o', 'n', 'e', '1'], ['t', 'w', 'o', '2']]

        >>> def cond(n: int) -> bool:
        ...     return n % 3 == 0
        >>>
        >>> pc.Iter.from_(range(10)).split_after(cond).into(list)
        [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> pc.Iter.from_(range(10)).split_after(cond, max_split=2).into(list)
        [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9]]

        ```
        """
        return self._lazy(mit.split_after, predicate, max_split)

    def split_before(
        self, predicate: Callable[[T], bool], max_split: int = -1
    ) -> Iter[list[T]]:
        """
        Yield lists of items from iterable, where each list ends with an item where callable pred returns True.

        Args:
            predicate: Function to determine the split points.
            max_split: Maximum number of splits to perform. Defaults to -1 (no limit).
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_("abcdcba").split_before(lambda x: x == "b").into(list)
        [['a'], ['b', 'c', 'd', 'c'], ['b', 'a']]
        >>>
        >>> def cond(n: int) -> bool:
        ...     return n % 2 == 1
        >>>
        >>> pc.Iter.from_(range(10)).split_before(cond).into(list)
        [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9]]

        ```
        At most *max_split* splits are done.

        If *max_split* is not specified or -1, then there is no limit on the number of splits:
        ```python
        >>> pc.Iter.from_(range(10)).split_before(cond, max_split=2).into(list)
        [[0], [1, 2], [3, 4, 5, 6, 7, 8, 9]]

        ```
        """
        return self._lazy(mit.split_before, predicate, max_split)

    def split_into(self, sizes: Iterable[int | None]) -> Iter[list[T]]:
        """
        Yield a list of sequential items from iterable of length 'n' for each integer 'n' in sizes.

        Args:
            sizes: Iterable of integers specifying the sizes of each chunk. Use None for the remainder.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3, 4, 5, 6]).split_into([1, 2, 3]).into(list)
        [[1], [2, 3], [4, 5, 6]]

        If the sum of sizes is smaller than the length of iterable, then the remaining items of iterable will not be returned.
        ```python
        >>> pc.Iter.from_([1, 2, 3, 4, 5, 6]).split_into([2, 3]).into(list)
        [[1, 2], [3, 4, 5]]

        ```

        If the sum of sizes is larger than the length of iterable:

        - fewer items will be returned in the iteration that overruns the iterable
        - further lists will be empty
        ```python
        >>> pc.Iter.from_([1, 2, 3, 4]).split_into([1, 2, 3, 4]).into(list)
        [[1], [2, 3], [4], []]

        ```

        When a None object is encountered in sizes, the returned list will contain items up to the end of iterable the same way that itertools.slice does:
        ```python
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        >>> pc.Iter.from_(data).split_into([2, 3, None]).into(list)
        [[1, 2], [3, 4, 5], [6, 7, 8, 9, 0]]

        ```

        split_into can be useful for grouping a series of items where the sizes of the groups are not uniform.

        An example would be where in a row from a table:

        - multiple columns represent elements of the same feature (e.g. a point represented by x,y,z)
        - the format is not the same for all columns.
        """
        return self._lazy(mit.split_into, sizes)

    def split_when(
        self, predicate: Callable[[T, T], bool], max_split: int = -1
    ) -> Iter[list[T]]:
        """
        Split iterable into pieces based on the output of a predicate function.

        Args:
            predicate: Function that takes successive pairs of items and returns True if the iterable should be split.
            max_split: Maximum number of splits to perform. Defaults to -1 (no limit).

        For example, to find runs of increasing numbers, split the iterable when element i is larger than element i + 1:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Seq([1, 2, 3, 3, 2, 5, 2, 4, 2])
        >>> data.iter().split_when(lambda x, y: x > y).into(list)
        [[1, 2, 3, 3], [2, 5], [2, 4], [2]]

        ```

        At most max_split splits are done.

        If max_split is not specified or -1, then there is no limit on the number of splits:
        ```python
        >>> data.iter().split_when(lambda x, y: x > y, max_split=2).into(list)
        [[1, 2, 3, 3], [2, 5], [2, 4, 2]]

        ```
        """
        return self._lazy(mit.split_when, predicate, max_split)

    def chunks(self, n: int, strict: bool = False) -> Iter[list[T]]:
        """
        Break iterable into lists of length n.

        By default, the last yielded list will have fewer than *n* elements if the length of *iterable* is not divisible by *n*.

        To use a fill-in value instead, see the :func:`grouper` recipe.

        If:

        - the length of *iterable* is not divisible by *n*
        - *strict* is `True`

        then `ValueError` will be raised before the last list is yielded.

        Args:
            n: Number of elements in each chunk.
            strict: Whether to raise an error if the last chunk is smaller than n. Defaults to False.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_([1, 2, 3, 4, 5, 6]).chunks(3).into(list)
        [[1, 2, 3], [4, 5, 6]]
        >>> pc.Iter.from_([1, 2, 3, 4, 5, 6, 7, 8]).chunks(3).into(list)
        [[1, 2, 3], [4, 5, 6], [7, 8]]

        ```
        """
        return self._lazy(mit.chunked, n, strict)

    def chunks_even(self, n: int) -> Iter[list[T]]:
        """
        Break iterable into lists of approximately length n.

        Items are distributed such the lengths of the lists differ by at most 1 item.

        Args:
            n: Approximate number of elements in each chunk.
        Example:
        ```python
        >>> import pyochain as pc
        >>> iterable = pc.Seq([1, 2, 3, 4, 5, 6, 7])
        >>> iterable.iter().chunks_even(3).into(list)  # List lengths: 3, 2, 2
        [[1, 2, 3], [4, 5], [6, 7]]
        >>> iterable.iter().chunks(3).into(list)  # List lengths: 3, 3, 1
        [[1, 2, 3], [4, 5, 6], [7]]

        ```
        """
        return self._lazy(mit.chunked_even, n)

    def unique_to_each[U: Iterable[Any]](self: IterWrapper[U]) -> Iter[list[U]]:
        """
        Return the elements from each of the iterables that aren't in the other iterables.

        For example, suppose you have a set of packages, each with a set of dependencies:

        **{'pkg_1': {'A', 'B'}, 'pkg_2': {'B', 'C'}, 'pkg_3': {'B', 'D'}}**

        If you remove one package, which dependencies can also be removed?

        If pkg_1 is removed, then A is no longer necessary - it is not associated with pkg_2 or pkg_3.

        Similarly, C is only needed for pkg_2, and D is only needed for pkg_3:
        ```python
        >>> import pyochain as pc
        >>> data = ({"A", "B"}, {"B", "C"}, {"B", "D"})
        >>> pc.Iter.from_(data).unique_to_each().collect()
        Seq([['A'], ['C'], ['D']])

        ```

        If there are duplicates in one input iterable that aren't in the others they will be duplicated in the output.

        Input order is preserved:
        ```python
        >>> data = ("mississippi", "missouri")
        >>> pc.Iter.from_(data).unique_to_each().collect()
        Seq([['p', 'p'], ['o', 'u', 'r']])

        ```

        It is assumed that the elements of each iterable are hashable.
        """

        from collections import Counter

        def _unique_to_each(data: Iterable[U]) -> Generator[list[U], None, None]:
            """from more_itertools.unique_to_each"""
            pool: list[Iterable[U]] = [it for it in data]
            counts: Counter[U] = Counter(itertools.chain.from_iterable(map(set, pool)))
            uniques: set[U] = {element for element in counts if counts[element] == 1}
            return ((list(filter(uniques.__contains__, it))) for it in pool)

        return self._lazy(_unique_to_each)
