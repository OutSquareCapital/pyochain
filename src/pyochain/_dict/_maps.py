from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import Dict


class MapDict[K, V](MappingWrapper[K, V]):
    def map_keys[T](self, func: Callable[[K], T]) -> Dict[T, V]:
        """
        Return keys transformed by func.

        Args:
            func: Function to apply to each key in the dictionary.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": [20, 15, 30], "Bob": [10, 35]}).map_keys(
        ...     str.lower
        ... ).inner()
        {'alice': [20, 15, 30], 'bob': [10, 35]}
        >>>
        >>> pc.Dict({1: "a"}).map_keys(str).inner()
        {'1': 'a'}

        ```
        """
        return self._new(partial(cz.dicttoolz.keymap, func))

    def map_values[T](self, func: Callable[[V], T]) -> Dict[K, T]:
        """
        Return values transformed by func.

        Args:
            func: Function to apply to each value in the dictionary.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": [20, 15, 30], "Bob": [10, 35]}).map_values(sum).inner()
        {'Alice': 65, 'Bob': 45}
        >>>
        >>> pc.Dict({1: 1}).map_values(lambda v: v + 1).inner()
        {1: 2}

        ```
        """
        return self._new(partial(cz.dicttoolz.valmap, func))

    def map_items[KR, VR](
        self,
        func: Callable[[tuple[K, V]], tuple[KR, VR]],
    ) -> Dict[KR, VR]:
        """
        Transform (key, value) pairs using a function that takes a (key, value) tuple.

        Args:
            func: Function to transform each (key, value) pair into a new (key, value) tuple.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": 10, "Bob": 20}).map_items(
        ...     lambda kv: (kv[0].upper(), kv[1] * 2)
        ... ).inner()
        {'ALICE': 20, 'BOB': 40}

        ```
        """
        return self._new(partial(cz.dicttoolz.itemmap, func))

    def map_kv[KR, VR](
        self,
        func: Callable[[K, V], tuple[KR, VR]],
    ) -> Dict[KR, VR]:
        """
        Transform (key, value) pairs using a function that takes key and value as separate arguments.

        Args:
            func: Function to transform each key and value into a new (key, value) tuple.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).map_kv(lambda k, v: (k + 1, v * 10)).inner()
        {2: 20}

        ```
        """

        def _map_kv(data: dict[K, V]) -> dict[KR, VR]:
            def _(kv: tuple[K, V]) -> tuple[KR, VR]:
                return func(kv[0], kv[1])

            return cz.dicttoolz.itemmap(_, data)

        return self._new(_map_kv)

    def invert(self) -> Dict[V, list[K]]:
        """
        Invert the dictionary, grouping keys by common (and hashable) values.
        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 1}
        >>> pc.Dict(d).invert().inner()
        {1: ['a', 'c'], 2: ['b']}

        ```
        """

        def _invert(data: dict[K, V]) -> dict[V, list[K]]:
            inverted: dict[V, list[K]] = defaultdict(list)
            for k, v in data.items():
                inverted[v].append(k)
            return dict(inverted)

        return self._new(_invert)

    def implode(self) -> Dict[K, list[V]]:
        """
        Nest all the values in lists.
        syntactic sugar for map_values(lambda v: [v])
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2, 3: 4}).implode().inner()
        {1: [2], 3: [4]}

        ```
        """

        def _implode(data: dict[K, V]) -> dict[K, list[V]]:
            def _(v: V) -> list[V]:
                return [v]

            return cz.dicttoolz.valmap(_, data)

        return self._new(_implode)
