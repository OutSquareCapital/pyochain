from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, TypeIs

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import Dict


def _drop_nones(
    data: dict[Any, Any] | list[Any], remove_empty: bool = True
) -> dict[Any, Any] | list[Any] | None:
    match data:
        case dict():
            pruned_dict: dict[Any, Any] = {}
            for k, v in data.items():
                pruned_v = _drop_nones(v, remove_empty)

                is_empty = remove_empty and (pruned_v is None or pruned_v in ({}, []))
                if not is_empty:
                    pruned_dict[k] = pruned_v
            return pruned_dict if pruned_dict or not remove_empty else None

        case list():
            pruned_list = [_drop_nones(item, remove_empty) for item in data]
            if remove_empty:
                pruned_list = [
                    item
                    for item in pruned_list
                    if not (item is None or item in ({}, []))
                ]
            return pruned_list if pruned_list or not remove_empty else None

        case _:
            if remove_empty and data is None:
                return None
            return data


class NestedDict[K, V](MappingWrapper[K, V]):
    def struct[**P, R, U: dict[Any, Any]](
        self: NestedDict[K, U],
        func: Callable[Concatenate[Dict[K, U], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Dict[K, R]:
        """
        Apply a function to each value after wrapping it in a Dict.

        Args:
            func: Function to apply to each value after wrapping it in a Dict.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Syntactic sugar for `map_values(lambda data: func(pc.Dict(data), *args, **kwargs))`
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "person1": {"name": "Alice", "age": 30, "city": "New York"},
        ...     "person2": {"name": "Bob", "age": 25, "city": "Los Angeles"},
        ... }
        >>> pc.Dict(data).struct(lambda d: d.map_keys(str.upper).drop("AGE").inner())
        ... # doctest: +NORMALIZE_WHITESPACE
        {'person1': {'CITY': 'New York', 'NAME': 'Alice'},
        'person2': {'CITY': 'Los Angeles', 'NAME': 'Bob'}}

        ```
        """
        from ._main import Dict

        def _struct(data: Mapping[K, U]) -> dict[K, R]:
            def _(v: dict[Any, Any]) -> R:
                return func(Dict(v), *args, **kwargs)

            return cz.dicttoolz.valmap(_, data)

        return self._new(_struct)

    def flatten(
        self: NestedDict[str, Any], sep: str = ".", max_depth: int | None = None
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary, concatenating keys with the specified separator.

        Args:
            sep: Separator to use when concatenating keys
            max_depth: Maximum depth to flatten. If None, flattens completely.
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "config": {"params": {"retries": 3, "timeout": 30}, "mode": "fast"},
        ...     "version": 1.0,
        ... }
        >>> pc.Dict(data).flatten().inner()
        {'config.params.retries': 3, 'config.params.timeout': 30, 'config.mode': 'fast', 'version': 1.0}
        >>> pc.Dict(data).flatten(sep="_").inner()
        {'config_params_retries': 3, 'config_params_timeout': 30, 'config_mode': 'fast', 'version': 1.0}
        >>> pc.Dict(data).flatten(max_depth=1).inner()
        {'config.params': {'retries': 3, 'timeout': 30}, 'config.mode': 'fast', 'version': 1.0}

        ```
        """

        def _flatten(
            d: Mapping[Any, Any], parent_key: str = "", current_depth: int = 1
        ) -> dict[str, Any]:
            def _can_recurse(v: object) -> TypeIs[Mapping[Any, Any]]:
                return isinstance(v, Mapping) and (
                    max_depth is None or current_depth < max_depth + 1
                )

            items: list[tuple[str, Any]] = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if _can_recurse(v):
                    items.extend(_flatten(v, new_key, current_depth + 1).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return self._new(_flatten)

    def unpivot(
        self: NestedDict[str, Mapping[str, Any]],
    ) -> Dict[str, dict[str, Any]]:
        """
        Unpivot a nested dictionary by swapping rows and columns.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "row1": {"col1": "A", "col2": "B"},
        ...     "row2": {"col1": "C", "col2": "D"},
        ... }
        >>> pc.Dict(data).unpivot()
        ... # doctest: +NORMALIZE_WHITESPACE
        {'col1': {'row1': 'A', 'row2': 'C'}, 'col2': {'row1': 'B', 'row2': 'D'}}
        """

        def _unpivot(
            data: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for rkey, inner in data.items():
                for ckey, val in inner.items():
                    out.setdefault(ckey, {})[rkey] = val
            return out

        return self._new(_unpivot)

    def with_nested_key(self, *keys: K, value: V) -> Dict[K, V]:
        """
        Set a nested key path and return a new Dict with new, potentially nested, key value pair.

        Args:
            *keys: Sequence of keys representing the nested path.
            value: Value to set at the specified nested path.
        ```python
        >>> import pyochain as pc
        >>> purchase = {
        ...     "name": "Alice",
        ...     "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        ...     "credit card": "5555-1234-1234-1234",
        ... }
        >>> pc.Dict(purchase).with_nested_key(
        ...     "order", "costs", value=[0.25, 1.00]
        ... ).inner()
        {'name': 'Alice', 'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.0]}, 'credit card': '5555-1234-1234-1234'}

        ```
        """

        def _with_nested_key(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.assoc_in(data, keys, value=value)

        return self._new(_with_nested_key)

    def pluck[U: str | int](self: NestedDict[U, Any], *keys: str) -> Dict[U, Any]:
        """
        Extract values from nested dictionaries using a sequence of keys.

        Args:
            *keys: Sequence of keys to extract values from the nested dictionaries.
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "person1": {"name": "Alice", "age": 30},
        ...     "person2": {"name": "Bob", "age": 25},
        ... }
        >>> pc.Dict(data).pluck("name").inner()
        {'person1': 'Alice', 'person2': 'Bob'}

        ```
        """

        getter = partial(cz.dicttoolz.get_in, keys)

        def _pluck(data: Mapping[U, Any]) -> dict[U, Any]:
            return cz.dicttoolz.valmap(getter, data)

        return self._new(_pluck)

    def get_in(self, *keys: K, default: Any = None) -> Any:
        """
        Retrieve a value from a nested dictionary structure.

        Args:
            *keys: Sequence of keys representing the nested path to retrieve the value.
            default: Default value to return if the keys do not exist.

        ```python
        >>> import pyochain as pc
        >>> data = {"a": {"b": {"c": 1}}}
        >>> pc.Dict(data).get_in("a", "b", "c")
        1
        >>> pc.Dict(data).get_in("a", "x", default="Not Found")
        'Not Found'

        ```
        """

        def _get_in(data: Mapping[K, V]) -> Any:
            return cz.dicttoolz.get_in(keys, data, default)

        return self.into(_get_in)

    def drop_nones(self, remove_empty: bool = True) -> Dict[K, V]:
        """
        Recursively drop None values from the dictionary.

        Options to also remove empty dicts and lists.

        Args:
            remove_empty: If True (default), removes `None`, `{}` and `[]`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "a": 1,
        ...     "b": None,
        ...     "c": {},
        ...     "d": [],
        ...     "e": {"f": None, "g": 2},
        ...     "h": [1, None, {}],
        ...     "i": 0,
        ... }
        >>> p_data = pc.Dict(data)
        >>>
        >>> p_data.drop_nones().inner()
        {'a': 1, 'e': {'g': 2}, 'h': [1], 'i': 0}
        >>>
        >>> p_data.drop_nones().inner()
        {'a': 1, 'e': {'g': 2}, 'h': [1], 'i': 0}
        >>>
        >>> p_data.drop_nones(remove_empty=False).inner()
        {'a': 1, 'b': None, 'c': {}, 'd': [], 'e': {'f': None, 'g': 2}, 'h': [1, None, {}], 'i': 0}

        ```
        """

        def _apply_drop_nones(data: dict[K, V]) -> dict[Any, Any]:
            result = _drop_nones(data, remove_empty)
            return result if isinstance(result, dict) else dict()

        return self._new(_apply_drop_nones)
