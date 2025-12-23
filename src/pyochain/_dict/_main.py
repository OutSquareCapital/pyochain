from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeIs, overload

import cytoolz as cz

from .._core import CommonBase, get_config

if TYPE_CHECKING:
    from .._core import SupportsKeysAndGetItem
    from .._iter import Iter
    from .._results import Option


class Dict[K, V](CommonBase[dict[K, V]], Mapping[K, V]):
    """Wrapper for Python dictionaries with chainable methods."""

    __slots__ = ("_inner",)

    _inner: dict[K, V]

    def __repr__(self) -> str:
        return f"{self.into(lambda d: get_config().dict_repr(d._inner))}"

    def __iter__(self) -> Iterator[K]:
        return self._inner.__iter__()

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, key: K) -> V:
        return self._inner[key]

    def _new[KU, VU](self, func: Callable[[dict[K, V]], dict[KU, VU]]) -> Dict[KU, VU]:
        return Dict(func(self._inner))

    @staticmethod
    def from_[G, I](
        data: Mapping[G, I] | Iterable[tuple[G, I]] | SupportsKeysAndGetItem[G, I],
    ) -> Dict[G, I]:
        """Create a `Dict` from a convertible value.

        Args:
            data (Mapping[G, I] | Iterable[tuple[G, I]] | SupportsKeysAndGetItem[G, I]): Object convertible into a Dict.

        Returns:
            Dict[G, I]: Instance containing the data from the input.

        Example:
        ```python
        >>> import pyochain as pc
        >>> class MyMapping:
        ...     def __init__(self):
        ...         self._data = {1: "a", 2: "b", 3: "c"}
        ...
        ...     def keys(self):
        ...         return self._data.keys()
        ...
        ...     def __getitem__(self, key):
        ...         return self._data[key]
        >>>
        >>> pc.Dict.from_(MyMapping())
        {1: 'a', 2: 'b', 3: 'c'}
        >>> pc.Dict.from_([("d", "e"), ("f", "g")])
        {'d': 'e', 'f': 'g'}

        ```
        """
        return Dict(dict(data))

    @staticmethod
    def from_object(obj: object) -> Dict[str, Any]:
        """Create a `Dict` from an object `__dict__` attribute.

        We can't know in advance the values types, so we use `Any`.

        Args:
            obj (object): The object whose `__dict__` attribute will be used to create the Dict.

        Returns:
            Dict[str, Any]: A new Dict instance containing the attributes of the object.

        ```python
        >>> import pyochain as pc
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Alice", 30)
        >>> pc.Dict.from_object(person)
        {'name': 'Alice', 'age': 30}

        ```
        """
        return Dict(obj.__dict__)

    def iter_values(self) -> Iter[V]:
        """Return an Iter of the dict's values.

        Returns:
            Iter[V]: An Iter wrapping the dictionary's values.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).iter_values().collect()
        Seq(2,)

        ```
        """
        from .._iter import Iter

        return self.into(lambda d: Iter(d._inner.values()))

    def iter_items(self) -> Iter[tuple[K, V]]:
        """Return an Iter of the dict's items.

        Returns:
            Iter[tuple[K, V]]: An Iter wrapping the dictionary's (key, value) pairs.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"a": 1, "b": 2}).iter_items().collect()
        Seq(('a', 1), ('b', 2))

        ```
        """
        from .._iter import Iter

        return self.into(lambda d: Iter(d._inner.items()))

    def map_keys[T](self, func: Callable[[K], T]) -> Dict[T, V]:
        """Return keys transformed by func.

        Args:
            func (Callable[[K], T]): Function to apply to each key in the dictionary.

        Returns:
            Dict[T, V]: Dict with transformed keys.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": [20, 15, 30], "Bob": [10, 35]}).map_keys(str.lower)
        {'alice': [20, 15, 30], 'bob': [10, 35]}
        >>>
        >>> pc.Dict({1: "a"}).map_keys(str)
        {'1': 'a'}

        ```
        """
        return self._new(partial(cz.dicttoolz.keymap, func))

    def map_values[T](self, func: Callable[[V], T]) -> Dict[K, T]:
        """Return values transformed by func.

        Args:
            func (Callable[[V], T]): Function to apply to each value in the dictionary.

        Returns:
            Dict[K, T]: Dict with transformed values.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": [20, 15, 30], "Bob": [10, 35]}).map_values(sum)
        {'Alice': 65, 'Bob': 45}
        >>>
        >>> pc.Dict({1: 1}).map_values(lambda v: v + 1)
        {1: 2}

        ```
        """
        return self._new(partial(cz.dicttoolz.valmap, func))

    def map_items[KR, VR](
        self,
        func: Callable[[tuple[K, V]], tuple[KR, VR]],
    ) -> Dict[KR, VR]:
        """Transform (key, value) pairs using a function that takes a (key, value) tuple.

        Args:
            func (Callable[[tuple[K, V]], tuple[KR, VR]]): Function to transform each (key, value) pair into a new (key, value) tuple.

        Returns:
            Dict[KR, VR]: Dict with transformed items.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"Alice": 10, "Bob": 20}).map_items(
        ...     lambda kv: (kv[0].upper(), kv[1] * 2)
        ... )
        {'ALICE': 20, 'BOB': 40}

        ```
        """
        return self._new(partial(cz.dicttoolz.itemmap, func))

    def update_in(
        self,
        *keys: K,
        func: Callable[[V], V],
        default: V | None = None,
    ) -> Dict[K, V]:
        """Update value in a (potentially) nested dictionary.

        Args:
            *keys (K): keys representing the nested path to update.
            func (Callable[[V], V]): Function to apply to the value at the specified path.
            default (V | None): Default value to use if the path does not exist, by default None

        Returns:
            Dict[K, V]: Dict with the updated value at the nested path.

        Applies the func to the value at the path specified by keys, returning a new Dict with the updated value.

        If the path does not exist, it will be created with the default value (if provided) before applying func.
        ```python
        >>> import pyochain as pc
        >>> inc = lambda x: x + 1
        >>> pc.Dict({"a": 0}).update_in("a", func=inc)
        {'a': 1}
        >>> transaction = {
        ...     "name": "Alice",
        ...     "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        ...     "credit card": "5555-1234-1234-1234",
        ... }
        >>> pc.Dict(transaction).update_in("purchase", "costs", func=sum) # doctest: +NORMALIZE_WHITESPACE
        {'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': 1.75},
        'credit card': '5555-1234-1234-1234'}

        >>> # updating a value when k0 is not in d
        >>> pc.Dict({}).update_in(1, 2, 3, func=str, default="bar")
        {1: {2: {3: 'bar'}}}
        >>> pc.Dict({1: "foo"}).update_in(2, 3, 4, func=inc, default=0)
        {1: 'foo', 2: {3: {4: 1}}}

        ```
        """

        def _update_in(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.update_in(data, keys, func, default=default)

        return self._new(_update_in)

    def drop(self, *keys: K) -> Dict[K, V]:
        """Return a new Dict with given keys removed.

        Args:
            *keys (K): keys to remove from the dictionary.

        Returns:
            Dict[K, V]: New Dict with specified keys removed.

        New dict has d[key] deleted for each supplied key.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"x": 1, "y": 2}).drop("y")
        {'x': 1}
        >>> pc.Dict({"x": 1, "y": 2}).drop("y", "x")
        {}
        >>> pc.Dict({"x": 1}).drop("y")  # Ignores missing keys
        {'x': 1}
        >>> pc.Dict({1: 2, 3: 4}).drop(1)
        {3: 4}

        ```
        """

        def _drop(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.dissoc(data, *keys)

        return self._new(_drop)

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

    def get_in(self, *keys: K) -> Option[V]:
        """Retrieve a value from a nested dictionary structure.

        Args:
            *keys (K): keys representing the nested path to retrieve the value.

        Returns:
            Option[V]: Value at the nested path or default if not found.

        ```python
        >>> import pyochain as pc
        >>> data = {"a": {"b": {"c": 1}}}
        >>> pc.Dict(data).get_in("a", "b", "c")
        Some(1)
        >>> pc.Dict(data).get_in("a", "x").unwrap_or('Not Found')
        'Not Found'

        ```
        """
        from .._results import Option

        def _get_in(data: Mapping[K, V]) -> Option[V]:
            return Option.from_(cz.dicttoolz.get_in(keys, data, None))

        return self.into(lambda d: _get_in(d._inner))
