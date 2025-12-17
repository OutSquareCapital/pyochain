from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, TypeIs, overload

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import Dict


class FilterDict[K, V](MappingWrapper[K, V]):
    @overload
    def filter_keys[U](self, predicate: Callable[[K], TypeIs[U]]) -> Dict[U, V]: ...
    @overload
    def filter_keys(self, predicate: Callable[[K], bool]) -> Dict[K, V]: ...
    def filter_keys[U](
        self,
        predicate: Callable[[K], bool | TypeIs[U]],
    ) -> Dict[K, V] | Dict[U, V]:
        """Return keys that satisfy predicate.

        Args:
            predicate (Callable[[K], bool | TypeIs[U]]): Function to determine if a key should be included.

        Returns:
            Dict[K, V] | Dict[U, V]: Filtered Dict with keys satisfying predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
        >>> pc.Dict(d).filter_keys(lambda x: x % 2 == 0)
        {2: 3, 4: 5}

        ```
        """
        return self._new(partial(cz.dicttoolz.keyfilter, predicate))

    @overload
    def filter_values[U](self, predicate: Callable[[V], TypeIs[U]]) -> Dict[K, U]: ...
    @overload
    def filter_values(self, predicate: Callable[[V], bool]) -> Dict[K, V]: ...
    def filter_values[U](
        self,
        predicate: Callable[[V], bool] | Callable[[V], TypeIs[U]],
    ) -> Dict[K, V] | Dict[K, U]:
        """Return items whose values satisfy predicate.

        Args:
            predicate (Callable[[V], bool] | Callable[[V], TypeIs[U]]): Function to determine if a value should be included.

        Returns:
            Dict[K, V] | Dict[K, U]: Filtered Dict with values satisfying predicate

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
        >>> pc.Dict(d).filter_values(lambda x: x % 2 == 0)
        {1: 2, 3: 4}
        >>> pc.Dict(d).filter_values(lambda x: not x > 3)
        {1: 2, 2: 3}

        ```
        """
        return self._new(partial(cz.dicttoolz.valfilter, predicate))

    def filter_items(self, predicate: Callable[[tuple[K, V]], bool]) -> Dict[K, V]:
        """Filter items by predicate applied to (key, value) tuples.

        Args:
            predicate (Callable[[tuple[K, V]], bool]): Function to determine if a (key, value) pair should be included.

        Returns:
            Dict[K, V]: A new Dict instance containing only the items that satisfy the predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def isvalid(item):
        ...     k, v = item
        ...     return k % 2 == 0 and v < 4
        >>> d = pc.Dict({1: 2, 2: 3, 3: 4, 4: 5})
        >>>
        >>> d.filter_items(isvalid)
        {2: 3}
        >>> d.filter_items(lambda kv: not isvalid(kv))
        {1: 2, 3: 4, 4: 5}

        ```
        """
        return self._new(partial(cz.dicttoolz.itemfilter, predicate))

    def filter_kv(self, predicate: Callable[[K, V], bool]) -> Dict[K, V]:
        """Filter items by predicate applied to unpacked (key, value) tuples.

        Args:
            predicate(Callable[[K, V], bool]): Function to determine if a key-value pair should be included.

        Returns:
            Dict[K, V]: Filtered Dict with items satisfying predicate.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def isvalid(key, value):
        ...     return key % 2 == 0 and value < 4
        >>> d = pc.Dict({1: 2, 2: 3, 3: 4, 4: 5})
        >>>
        >>> d.filter_kv(isvalid)
        {2: 3}
        >>> d.filter_kv(lambda k, v: not isvalid(k, v))
        {1: 2, 3: 4, 4: 5}

        ```
        """

        def _filter_kv(data: dict[K, V]) -> dict[K, V]:
            def _(kv: tuple[K, V]) -> bool:
                return predicate(kv[0], kv[1])

            return cz.dicttoolz.itemfilter(_, data)

        return self._new(_filter_kv)

    def intersect_keys(self, *others: Mapping[K, V]) -> Dict[K, V]:
        """Keep only keys present in self and all others mappings.

        Args:
            *others (Mapping[K, V]): Other mappings to intersect keys with.

        Returns:
            Dict[K, V]: A new Dict with only the intersected keys.

        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"b": 10, "c": 20}
        >>> d3 = {"c": 30}
        >>> pc.Dict(d1).intersect_keys(d2, d3)
        {'c': 3}

        ```
        """

        def _intersect_keys(data: dict[K, V]) -> dict[K, V]:
            self_keys = set(data.keys())
            for other in others:
                self_keys.intersection_update(other.keys())
            return {k: data[k] for k in self_keys}

        return self._new(_intersect_keys)

    def diff_keys(self, *others: Mapping[K, V]) -> Dict[K, V]:
        """Keep only keys present in self but not in others mappings.

        Args:
            *others (Mapping[K, V]): Other mappings to exclude keys from.

        Returns:
            Dict[K, V]: A new Dict with only the differing keys.
        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"b": 10, "d": 40}
        >>> d3 = {"c": 30}
        >>> pc.Dict(d1).diff_keys(d2, d3)
        {'a': 1}

        ```
        """

        def _diff_keys(data: dict[K, V]) -> dict[K, V]:
            self_keys = set(data.keys())
            for other in others:
                self_keys.difference_update(other.keys())
            return {k: data[k] for k in self_keys}

        return self._new(_diff_keys)
