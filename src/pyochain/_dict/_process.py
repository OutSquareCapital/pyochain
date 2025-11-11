from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Concatenate

import cytoolz as cz

from .._core import MappingWrapper, SupportsRichComparison

if TYPE_CHECKING:
    from ._main import Dict


class ProcessDict[K, V](MappingWrapper[K, V]):
    def for_each[**P](
        self,
        func: Callable[Concatenate[K, V, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Dict[K, V]:
        """
        Apply a function to each key-value pair in the dict for side effects.

        Args:
            func: Function to apply to each key-value pair.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns the original Dict unchanged.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"a": 1, "b": 2}).for_each(
        ...     lambda k, v: print(f"Key: {k}, Value: {v}")
        ... ).inner()
        Key: a, Value: 1
        Key: b, Value: 2
        {'a': 1, 'b': 2}

        ```
        """

        def _for_each(data: dict[K, V]) -> dict[K, V]:
            for k, v in data.items():
                func(k, v, *args, **kwargs)
            return data

        return self._new(_for_each)

    def update_in(
        self, *keys: K, func: Callable[[V], V], default: V | None = None
    ) -> Dict[K, V]:
        """
        Update value in a (potentially) nested dictionary.

        Args:
            *keys: Sequence of keys representing the nested path to update.
            func: Function to apply to the value at the specified path.
            default: Default value to use if the path does not exist, by default None

        Applies the func to the value at the path specified by keys, returning a new Dict with the updated value.

        If the path does not exist, it will be created with the default value (if provided) before applying func.
        ```python
        >>> import pyochain as pc
        >>> inc = lambda x: x + 1
        >>> pc.Dict({"a": 0}).update_in("a", func=inc).inner()
        {'a': 1}
        >>> transaction = {
        ...     "name": "Alice",
        ...     "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        ...     "credit card": "5555-1234-1234-1234",
        ... }
        >>> pc.Dict(transaction).update_in("purchase", "costs", func=sum).inner()
        {'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': 1.75}, 'credit card': '5555-1234-1234-1234'}
        >>> # updating a value when k0 is not in d
        >>> pc.Dict({}).update_in(1, 2, 3, func=str, default="bar").inner()
        {1: {2: {3: 'bar'}}}
        >>> pc.Dict({1: "foo"}).update_in(2, 3, 4, func=inc, default=0).inner()
        {1: 'foo', 2: {3: {4: 1}}}

        ```
        """

        def _update_in(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.update_in(data, keys, func, default=default)

        return self._new(_update_in)

    def with_key(self, key: K, value: V) -> Dict[K, V]:
        """
        Return a new Dict with key set to value.

        Args:
            key: Key to set in the dictionary.
            value: Value to associate with the specified key.

        Does not modify the initial dictionary.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"x": 1}).with_key("x", 2).inner()
        {'x': 2}
        >>> pc.Dict({"x": 1}).with_key("y", 3).inner()
        {'x': 1, 'y': 3}
        >>> pc.Dict({}).with_key("x", 1).inner()
        {'x': 1}

        ```
        """

        def _with_key(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.assoc(data, key, value)

        return self._new(_with_key)

    def drop(self, *keys: K) -> Dict[K, V]:
        """
        Return a new Dict with given keys removed.

        Args:
            *keys: Sequence of keys to remove from the dictionary.

        New dict has d[key] deleted for each supplied key.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"x": 1, "y": 2}).drop("y").inner()
        {'x': 1}
        >>> pc.Dict({"x": 1, "y": 2}).drop("y", "x").inner()
        {}
        >>> pc.Dict({"x": 1}).drop("y").inner()  # Ignores missing keys
        {'x': 1}
        >>> pc.Dict({1: 2, 3: 4}).drop(1).inner()
        {3: 4}

        ```
        """

        def _drop(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.dissoc(data, *keys)

        return self._new(_drop)

    def rename(self, mapping: Mapping[K, K]) -> Dict[K, V]:
        """
        Return a new Dict with keys renamed according to the mapping.

        Args:
            mapping: A dictionary mapping old keys to new keys.

        Keys not in the mapping are kept as is.
        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 3}
        >>> mapping = {"b": "beta", "c": "gamma"}
        >>> pc.Dict(d).rename(mapping).inner()
        {'a': 1, 'beta': 2, 'gamma': 3}

        ```
        """

        def _rename(data: dict[K, V]) -> dict[K, V]:
            return {mapping.get(k, k): v for k, v in data.items()}

        return self._new(_rename)

    def sort(self, reverse: bool = False) -> Dict[K, V]:
        """
        Sort the dictionary by its keys and return a new Dict.

        Args:
            reverse: Whether to sort in descending order. Defaults to False.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"b": 2, "a": 1}).sort().inner()
        {'a': 1, 'b': 2}

        ```
        """

        def _sort(data: dict[K, V]) -> dict[K, V]:
            return dict(sorted(data.items(), reverse=reverse))

        return self._new(_sort)

    def sort_values[U: SupportsRichComparison[Any]](
        self: ProcessDict[K, U], reverse: bool = False
    ) -> Dict[K, U]:
        """
        Sort the dictionary by its values and return a new Dict.

        Args:
            reverse: Whether to sort in descending order. Defaults to False.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"a": 2, "b": 1}).sort_values().inner()
        {'b': 1, 'a': 2}

        ```
        """

        def _sort_values(data: dict[K, U]) -> dict[K, U]:
            return dict(sorted(data.items(), key=lambda item: item[1], reverse=reverse))

        return self._new(_sort_values)
