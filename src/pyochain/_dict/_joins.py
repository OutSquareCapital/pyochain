from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import Dict


class JoinsDict[K, V](MappingWrapper[K, V]):
    def inner_join[W](self, other: Mapping[K, W]) -> Dict[K, tuple[V, W]]:
        """Performs an inner join with another mapping based on keys.

        Args:
            other(Mapping[K, W]): The mapping to join with.

        Returns:
            Dict[K, tuple[V, W]]: Joined Dict with tuples of values from both mappings.

        Only keys present in both mappings are kept.
        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2}
        >>> d2 = {"b": 10, "c": 20}
        >>> pc.Dict(d1).inner_join(d2).inner()
        {'b': (2, 10)}

        ```
        """

        def _inner_join(data: Mapping[K, V]) -> dict[K, tuple[V, W]]:
            return {k: (v, other[k]) for k, v in data.items() if k in other}

        return self._new(_inner_join)

    def left_join[W](self, other: Mapping[K, W]) -> Dict[K, tuple[V, W | None]]:
        """Performs a left join with another mapping based on keys.

        Args:
            other(Mapping[K, W]): The mapping to join with.

        Returns:
            Dict[K, tuple[V, W | None]]: Joined Dict with tuples of values, right side can be None.

        All keys from the left dictionary (self) are kept.
        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2}
        >>> d2 = {"b": 10, "c": 20}
        >>> pc.Dict(d1).left_join(d2).inner()
        {'a': (1, None), 'b': (2, 10)}

        ```
        """

        def _left_join(data: Mapping[K, V]) -> dict[K, tuple[V, W | None]]:
            return {k: (v, other.get(k)) for k, v in data.items()}

        return self._new(_left_join)

    def diff(self, other: Mapping[K, V]) -> Dict[K, tuple[V | None, V | None]]:
        """Returns a dict of the differences between this dict and another.

        Args:
            other(Mapping[K, V]): The mapping to compare against.

        Returns:
            Dict[K, tuple[V | None, V | None]]: Dict with differences as (self_value, other_value) tuples.

        The keys of the returned dict are the keys that are not shared or have different values.
        The values are tuples containing the value from self and the value from other.
        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"b": 2, "c": 4, "d": 5}
        >>> pc.Dict(d1).diff(d2).sort().inner()
        {'a': (1, None), 'c': (3, 4), 'd': (None, 5)}

        ```
        """

        def _diff(data: Mapping[K, V]) -> dict[K, tuple[V | None, V | None]]:
            all_keys: set[K] = data.keys() | other.keys()
            diffs: dict[K, tuple[V | None, V | None]] = {}
            for key in all_keys:
                self_val = data.get(key)
                other_val = other.get(key)
                if self_val != other_val:
                    diffs[key] = (self_val, other_val)
            return diffs

        return self._new(_diff)

    def merge(self, *others: Mapping[K, V]) -> Dict[K, V]:
        """Merge other dicts into this one.

        Args:
            *others(Mapping[K, V]): One or more mappings to merge into the current dictionary.

        Returns:
            Dict[K, V]: Merged Dict with values from all dicts.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: "one"}).merge({2: "two"}).inner()
        {1: 'one', 2: 'two'}
        >>> # Later dictionaries have precedence
        >>> pc.Dict({1: 2, 3: 4}).merge({3: 3, 4: 4}).inner()
        {1: 2, 3: 3, 4: 4}

        ```
        """

        def _merge(data: Mapping[K, V]) -> dict[K, V]:
            return cz.dicttoolz.merge(data, *others)

        return self._new(_merge)

    def merge_with(
        self,
        *others: Mapping[K, V],
        func: Callable[[Iterable[V]], V],
    ) -> Dict[K, V]:
        """Merge dicts using a function to combine values for duplicate keys.

        Args:
            *others(Mapping[K, V]): One or more mappings to merge into the current dictionary.
            func(Callable[[Iterable[V]], V]): Function to combine values for duplicate keys.

        Returns:
            Dict[K, V]: Merged Dict with combined values.

        A key may occur in more than one dict, and all values mapped from the key will be passed to the function as a list, such as func([val1, val2, ...]).
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 1, 2: 2}).merge_with({1: 10, 2: 20}, func=sum).inner()
        {1: 11, 2: 22}
        >>> pc.Dict({1: 1, 2: 2}).merge_with({2: 20, 3: 30}, func=max).inner()
        {1: 1, 2: 20, 3: 30}

        ```
        """

        def _merge_with(data: Mapping[K, V]) -> dict[K, V]:
            return cz.dicttoolz.merge_with(func, data, *others)

        return self._new(_merge_with)
