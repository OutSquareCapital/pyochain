from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import Dict


class GroupsDict[K, V](MappingWrapper[K, V]):
    def group_by_value[G](self, func: Callable[[V], G]) -> Dict[G, dict[K, V]]:
        """
        Group dict items into sub-dictionaries based on a function of the value.

        Args:
            func: Function to determine the group for each value.

        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 3, "d": 2}
        >>> pc.Dict(d).group_by_value(lambda v: v % 2).inner()
        {1: {'a': 1, 'c': 3}, 0: {'b': 2, 'd': 2}}

        ```
        """

        def _group_by_value(data: dict[K, V]) -> dict[G, dict[K, V]]:
            def _(kv: tuple[K, V]) -> G:
                return func(kv[1])

            return cz.dicttoolz.valmap(dict, cz.itertoolz.groupby(_, data.items()))

        return self._new(_group_by_value)

    def group_by_key[G](self, func: Callable[[K], G]) -> Dict[G, dict[K, V]]:
        """
        Group dict items into sub-dictionaries based on a function of the key.

        Args:
            func: Function to determine the group for each key.

        ```python
        >>> import pyochain as pc
        >>> d = {"user_1": 10, "user_2": 20, "admin_1": 100}
        >>> pc.Dict(d).group_by_key(lambda k: k.split("_")[0]).inner()
        {'user': {'user_1': 10, 'user_2': 20}, 'admin': {'admin_1': 100}}

        ```
        """

        def _group_by_key(data: dict[K, V]) -> dict[G, dict[K, V]]:
            def _(kv: tuple[K, V]) -> G:
                return func(kv[0])

            return cz.dicttoolz.valmap(dict, cz.itertoolz.groupby(_, data.items()))

        return self._new(_group_by_key)

    def group_by_key_agg[G, R](
        self,
        key_func: Callable[[K], G],
        agg_func: Callable[[Dict[K, V]], R],
    ) -> Dict[G, R]:
        """
        Group by key function, then apply aggregation function to each sub-dict.

        Args:
            key_func: Function to determine the group for each key.
            agg_func: Function to aggregate each sub-dictionary.

        This avoids materializing intermediate `dict` objects if you only need
        an aggregated result for each group.
        ```python
        >>> import pyochain as pc
        >>>
        >>> data = {"user_1": 10, "user_2": 20, "admin_1": 100}
        >>> pc.Dict(data).group_by_key_agg(
        ...     key_func=lambda k: k.split("_")[0],
        ...     agg_func=lambda d: d.iter_values().sum(),
        ... ).inner()
        {'user': 30, 'admin': 100}
        >>>
        >>> data_files = {
        ...     "file_a.txt": 100,
        ...     "file_b.log": 20,
        ...     "file_c.txt": 50,
        ...     "file_d.log": 5,
        ... }
        >>>
        >>> def get_stats(sub_dict: pc.Dict[str, int]) -> dict[str, Any]:
        ...     return {
        ...         "count": sub_dict.iter_keys().count(),
        ...         "total_size": sub_dict.iter_values().sum(),
        ...         "max_size": sub_dict.iter_values().max(),
        ...         "files": sub_dict.iter_keys().sort().into(list),
        ...     }
        >>>
        >>> pc.Dict(data_files).group_by_key_agg(
        ...     key_func=lambda k: k.split(".")[-1], agg_func=get_stats
        ... ).sort().inner()
        {'log': {'count': 2, 'total_size': 25, 'max_size': 20, 'files': ['file_b.log', 'file_d.log']}, 'txt': {'count': 2, 'total_size': 150, 'max_size': 100, 'files': ['file_a.txt', 'file_c.txt']}}

        ```
        """
        from ._main import Dict

        def _group_by_key_agg(data: dict[K, V]) -> dict[G, R]:
            def _key_func(kv: tuple[K, V]) -> G:
                return key_func(kv[0])

            def _agg_func(items: list[tuple[K, V]]) -> R:
                return agg_func(Dict(dict(items)))

            groups = cz.itertoolz.groupby(_key_func, data.items())
            return cz.dicttoolz.valmap(_agg_func, groups)

        return self._new(_group_by_key_agg)

    def group_by_value_agg[G, R](
        self,
        value_func: Callable[[V], G],
        agg_func: Callable[[Dict[K, V]], R],
    ) -> Dict[G, R]:
        """
        Group by value function, then apply aggregation function to each sub-dict.

        Args:
            value_func: Function to determine the group for each value.
            agg_func: Function to aggregate each sub-dictionary.

        This avoids materializing intermediate `dict` objects if you only need
        an aggregated result for each group.
        ```python
        >>> import pyochain as pc
        >>>
        >>> data = {"math": "A", "physics": "B", "english": "A"}
        >>> pc.Dict(data).group_by_value_agg(
        ...     value_func=lambda grade: grade,
        ...     agg_func=lambda d: d.iter_keys().count(),
        ... ).inner()
        {'A': 2, 'B': 1}
        >>> # Second example
        >>> sales_data = {
        ...     "store_1": "Electronics",
        ...     "store_2": "Groceries",
        ...     "store_3": "Electronics",
        ...     "store_4": "Clothing",
        ... }
        >>>
        >>> # Obtain the first store for each category (after sorting store names)
        >>> pc.Dict(sales_data).group_by_value_agg(
        ...     value_func=lambda category: category,
        ...     agg_func=lambda d: d.iter_keys().sort().first(),
        ... ).sort().inner()
        {'Clothing': 'store_4', 'Electronics': 'store_1', 'Groceries': 'store_2'}

        ```
        """
        from ._main import Dict

        def _group_by_value_agg(data: dict[K, V]) -> dict[G, R]:
            def _key_func(kv: tuple[K, V]) -> G:
                return value_func(kv[1])

            def _agg_func(items: list[tuple[K, V]]) -> R:
                return agg_func(Dict(dict(items)))

            groups = cz.itertoolz.groupby(_key_func, data.items())
            return cz.dicttoolz.valmap(_agg_func, groups)

        return self._new(_group_by_value_agg)
