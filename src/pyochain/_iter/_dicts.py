from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import cytoolz as cz

from .._core import IterWrapper

if TYPE_CHECKING:
    from .._dict import Dict


class BaseDict[T](IterWrapper[T]):
    def with_keys[K](self, keys: Iterable[K]) -> Dict[K, T]:
        """
        Create a Dict by zipping the iterable with keys.

        Args:
            keys: Iterable of keys to pair with the values.
        Example:
        ```python
        >>> import pyochain as pc
        >>> keys = ["a", "b", "c"]
        >>> values = [1, 2, 3]
        >>> pc.Iter.from_(values).with_keys(keys).inner()
        {'a': 1, 'b': 2, 'c': 3}
        >>> # This is equivalent to:
        >>> pc.Iter.from_(keys).zip(values).pipe(
        ...     lambda x: pc.Dict(x.into(dict)).inner()
        ... )
        {'a': 1, 'b': 2, 'c': 3}

        ```
        """
        from .._dict import Dict

        def _with_keys(data: Iterable[T]) -> Dict[K, T]:
            return Dict(dict(zip(keys, data)))

        return self.into(_with_keys)

    def with_values[V](self, values: Iterable[V]) -> Dict[T, V]:
        """
        Create a Dict by zipping the iterable with values.

        Args:
            values: Iterable of values to pair with the keys.
        Example:
        ```python
        >>> import pyochain as pc
        >>> keys = [1, 2, 3]
        >>> values = ["a", "b", "c"]
        >>> pc.Iter.from_(keys).with_values(values).inner()
        {1: 'a', 2: 'b', 3: 'c'}
        >>> # This is equivalent to:
        >>> pc.Iter.from_(keys).zip(values).pipe(
        ...     lambda x: pc.Dict(x.into(dict)).inner()
        ... )
        {1: 'a', 2: 'b', 3: 'c'}

        ```
        """
        from .._dict import Dict

        def _with_values(data: Iterable[T]) -> Dict[T, V]:
            return Dict(dict(zip(data, values)))

        return self.into(_with_values)

    def reduce_by[K](
        self, key: Callable[[T], K], binop: Callable[[T, T], T]
    ) -> Dict[K, T]:
        """
        Perform a simultaneous groupby and reduction.

        Args:
            key: Function to compute the key for grouping.
            binop: Binary operation to reduce the grouped elements.
        Example:
        ```python
        >>> from collections.abc import Iterable
        >>> import pyochain as pc
        >>> from operator import add, mul
        >>>
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>>
        >>> def group_reduce(data: Iterable[int]) -> int:
        ...     return pc.Iter.from_(data).reduce(add)
        >>>
        >>> data = pc.Seq([1, 2, 3, 4, 5])
        >>> data.iter().reduce_by(is_even, add).inner()
        {False: 9, True: 6}
        >>> data.iter().group_by(is_even).map_values(group_reduce).inner()
        {False: 9, True: 6}

        ```
        But the former does not build the intermediate groups, allowing it to operate in much less space.

        This makes it suitable for larger datasets that do not fit comfortably in memory

        Simple Examples:
        ```python
        >>> pc.Iter.from_([1, 2, 3, 4, 5]).reduce_by(is_even, add).inner()
        {False: 9, True: 6}
        >>> pc.Iter.from_([1, 2, 3, 4, 5]).reduce_by(is_even, mul).inner()
        {False: 15, True: 8}

        ```
        """
        from .._dict import Dict

        def _reduce_by(data: Iterable[T]) -> Dict[K, T]:
            return Dict(cz.itertoolz.reduceby(key, binop, data))

        return self.into(_reduce_by)

    def group_by[K](self, on: Callable[[T], K]) -> Dict[K, list[T]]:
        """
        Group elements by key function and return a Dict result.

        Args:
            on: Function to compute the key for grouping.
        Example:
        ```python
        >>> import pyochain as pc
        >>> names = [
        ...     "Alice",
        ...     "Bob",
        ...     "Charlie",
        ...     "Dan",
        ...     "Edith",
        ...     "Frank",
        ... ]
        >>> pc.Iter.from_(names).group_by(len).sort()
        ... # doctest: +NORMALIZE_WHITESPACE
        {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
        >>>
        >>> iseven = lambda x: x % 2 == 0
        >>> pc.Iter.from_([1, 2, 3, 4, 5, 6, 7, 8]).group_by(iseven)
        ... # doctest: +NORMALIZE_WHITESPACE
        {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

        ```
        Non-callable keys imply grouping on a member.
        ```python
        >>> data = [
        ...     {"name": "Alice", "gender": "F"},
        ...     {"name": "Bob", "gender": "M"},
        ...     {"name": "Charlie", "gender": "M"},
        ... ]
        >>> pc.Iter.from_(data).group_by("gender").sort()
        ... # doctest: +NORMALIZE_WHITESPACE
        {'F': [{'gender': 'F', 'name': 'Alice'}],
        'M': [{'gender': 'M', 'name': 'Bob'}, {'gender': 'M', 'name': 'Charlie'}]}

        ```
        """
        from .._dict import Dict

        def _group_by(data: Iterable[T]) -> Dict[K, list[T]]:
            return Dict(cz.itertoolz.groupby(on, data))

        return self.into(_group_by)

    def frequencies(self) -> Dict[T, int]:
        """
        Find number of occurrences of each value in the iterable.
        ```python
        >>> import pyochain as pc
        >>> data = ["cat", "cat", "ox", "pig", "pig", "cat"]
        >>> pc.Iter.from_(data).frequencies().inner()
        {'cat': 3, 'ox': 1, 'pig': 2}

        ```
        """
        from .._dict import Dict

        def _frequencies(data: Iterable[T]) -> Dict[T, int]:
            return Dict(cz.itertoolz.frequencies(data))

        return self.into(_frequencies)

    def count_by[K](self, key: Callable[[T], K]) -> Dict[K, int]:
        """
        Count elements of a collection by a key function.

        Args:
            key: Function to compute the key for counting.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(["cat", "mouse", "dog"]).count_by(len).inner()
        {3: 2, 5: 1}
        >>> def iseven(x):
        ...     return x % 2 == 0
        >>> pc.Iter.from_([1, 2, 3]).count_by(iseven).inner()
        {False: 2, True: 1}

        ```
        """
        from .._dict import Dict

        def _count_by(data: Iterable[T]) -> Dict[K, int]:
            return Dict(cz.recipes.countby(key, data))

        return self.into(_count_by)

    def to_records[U: Sequence[Any]](self: BaseDict[U]) -> Dict[Any, Any]:
        """
        Transform an iterable of nested sequences into a nested dictionary.

        - Each inner sequence represents a path to a value in the dictionary.
        - The last element of each sequence is treated as the value
        - All preceding elements are treated as keys leading to that value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> arrays = [["a", "b", 1], ["a", "c", 2], ["d", 3]]
        >>> pc.Seq(arrays).to_records().inner()
        {'a': {'b': 1, 'c': 2}, 'd': 3}

        ```
        """
        from .._dict import Dict

        def _from_nested(
            arrays: Iterable[Sequence[Any]], parent: dict[Any, Any] | None = None
        ) -> dict[Any, Any]:
            """from dictutils.pivot"""
            d: dict[Any, Any] = parent or {}
            for arr in arrays:
                if len(arr) >= 2:
                    head, *tail = arr
                    if len(tail) == 1:
                        d[head] = tail[0]
                    else:
                        d[head] = _from_nested([tail], d.get(head, {}))
            return d

        return Dict(self.into(_from_nested))
