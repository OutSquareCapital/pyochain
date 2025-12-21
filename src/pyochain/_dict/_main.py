from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeIs, overload

import cytoolz as cz

from .._core import CommonBase, get_config

if TYPE_CHECKING:
    from .._core import SupportsKeysAndGetItem
    from .._iter import Iter, Vec
    from .._results import Option


class Dict[K, V](CommonBase[dict[K, V]]):
    """Wrapper for Python dictionaries with chainable methods."""

    __slots__ = ("_inner",)

    _inner: dict[K, V]

    def __repr__(self) -> str:
        return f"{self.into(lambda d: get_config().dict_repr(d._inner))}"

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

    def pivot(self, *indices: int) -> Dict[Any, Any]:
        """Pivot a nested dictionary by rearranging the key levels according to order.

        Syntactic sugar for `Dict.to_arrays().rearrange(*indices).to_records()`

        Args:
            *indices (int): Indices specifying the new order of key levels

        Returns:
            Dict[Any, Any]: Pivoted dictionary with keys rearranged

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {"A": {"X": 1, "Y": 2}, "B": {"X": 3, "Y": 4}}
        >>> pc.Dict(d).pivot(1, 0)
        {'X': {'A': 1, 'B': 3}, 'Y': {'A': 2, 'B': 4}}
        """
        return self.to_arrays().rearrange(*indices).to_records()

    def iter_keys(self) -> Iter[K]:
        """Return an Iter of the dict's keys.

        Returns:
            Iter[K]: An Iter wrapping the dictionary's keys.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).iter_keys().collect()
        Seq(1,)

        ```
        """
        from .._iter import Iter

        return self.into(lambda d: Iter(d._inner.keys()))

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

    def to_arrays(self) -> Vec[list[Any]]:
        """Convert the nested dictionary into a `Vec` of arrays.

        The sequence represents all paths from root to leaves.

        Returns:
            Vec[list[Any]]: A `Vec` of arrays representing paths from root to leaves.

        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "a": {"b": 1, "c": 2},
        ...     "d": {"e": {"f": 3}},
        ... }
        >>> pc.Dict(data).to_arrays()
        Vec(['a', 'b', 1], ['a', 'c', 2], ['d', 'e', 'f', 3])

        ```
        """
        from .._iter import Vec

        def _to_arrays(d: Mapping[Any, Any]) -> list[list[Any]]:
            match d:
                case Mapping():
                    arr: list[Any] = []
                    for k, v in d.items():
                        arr.extend([[k, *el] for el in _to_arrays(v)])
                    return arr

                case _:
                    return [[d]]

        return self.into(lambda d: Vec(_to_arrays(d._inner)))

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

    def invert(self) -> Dict[V, list[K]]:
        """Invert the dictionary, grouping keys by common (and hashable) values.

        Returns:
            Dict[V, list[K]]: Inverted Dict with values as keys and lists of original keys as values.

        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 1}
        >>> pc.Dict(d).invert()
        {1: ['a', 'c'], 2: ['b']}

        ```
        """
        from collections import defaultdict

        def _invert(data: dict[K, V]) -> dict[V, list[K]]:
            inverted: dict[V, list[K]] = defaultdict(list)
            for k, v in data.items():
                inverted[v].append(k)
            return dict(inverted)

        return self._new(_invert)

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

    def with_key(self, key: K, value: V) -> Dict[K, V]:
        """Return a new Dict with key set to value.

        Args:
            key (K): Key to set in the dictionary.
            value (V): Value to associate with the specified key.

        Returns:
            Dict[K, V]: New Dict with the key-value pair set.

        Does not modify the initial dictionary.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"x": 1}).with_key("x", 2)
        {'x': 2}
        >>> pc.Dict({"x": 1}).with_key("y", 3)
        {'x': 1, 'y': 3}
        >>> pc.Dict({}).with_key("x", 1)
        {'x': 1}

        ```
        """

        def _with_key(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.assoc(data, key, value)

        return self._new(_with_key)

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

    def rename(self, mapping: Mapping[K, K]) -> Dict[K, V]:
        """Return a new Dict with keys renamed according to the mapping.

        Args:
            mapping (Mapping[K, K]): A dictionary mapping old keys to new keys.

        Returns:
            Dict[K, V]: Dict with keys renamed according to the mapping.

        Keys not in the mapping are kept as is.
        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 3}
        >>> mapping = {"b": "beta", "c": "gamma"}
        >>> pc.Dict(d).rename(mapping)
        {'a': 1, 'beta': 2, 'gamma': 3}

        ```
        """

        def _rename(data: dict[K, V]) -> dict[K, V]:
            return {mapping.get(k, k): v for k, v in data.items()}

        return self._new(_rename)

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
        >>> pc.Dict(d1).inner_join(d2)
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
        >>> pc.Dict(d1).left_join(d2)
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
        >>> pc.Dict(d1).diff(d2).iter_items().sort().into(pc.Dict.from_)
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
        >>> pc.Dict({1: "one"}).merge({2: "two"})
        {1: 'one', 2: 'two'}
        >>> # Later dictionaries have precedence
        >>> pc.Dict({1: 2, 3: 4}).merge({3: 3, 4: 4})
        {1: 2, 3: 3, 4: 4}

        ```
        """

        def _merge(data: Mapping[K, V]) -> dict[K, V]:
            return cz.dicttoolz.merge(data, *others)

        return self._new(_merge)

    def merge_with(
        self, *others: Mapping[K, V], func: Callable[[Iterable[V]], V]
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
        >>> pc.Dict({1: 1, 2: 2}).merge_with({1: 10, 2: 20}, func=sum)
        {1: 11, 2: 22}
        >>> pc.Dict({1: 1, 2: 2}).merge_with({2: 20, 3: 30}, func=max)
        {1: 1, 2: 20, 3: 30}

        ```
        """

        def _merge_with(data: Mapping[K, V]) -> dict[K, V]:
            return cz.dicttoolz.merge_with(func, data, *others)

        return self._new(_merge_with)

    def group_by_value[G](self, func: Callable[[V], G]) -> Dict[G, dict[K, V]]:
        """Group dict items into sub-dictionaries based on a function of the value.

        Args:
            func(Callable[[V], G]): Function to determine the group for each value.

        Returns:
            Dict[G, dict[K, V]]: Grouped Dict with groups as keys and sub-dicts as values.

        ```python
        >>> import pyochain as pc
        >>> d = {"a": 1, "b": 2, "c": 3, "d": 2}
        >>> pc.Dict(d).group_by_value(lambda v: v % 2)
        {1: {'a': 1, 'c': 3}, 0: {'b': 2, 'd': 2}}

        ```
        """

        def _group_by_value(data: dict[K, V]) -> dict[G, dict[K, V]]:
            def _(kv: tuple[K, V]) -> G:
                return func(kv[1])

            return cz.dicttoolz.valmap(dict, cz.itertoolz.groupby(_, data.items()))

        return self._new(_group_by_value)

    def group_by_key[G](self, func: Callable[[K], G]) -> Dict[G, dict[K, V]]:
        """Group dict items into sub-dictionaries based on a function of the key.

        Args:
            func(Callable[[K], G]): Function to determine the group for each key.

        Returns:
            Dict[G, dict[K, V]]: Grouped Dict with groups as keys and sub-dicts as values.

        ```python
        >>> import pyochain as pc
        >>> d = {"user_1": 10, "user_2": 20, "admin_1": 100}
        >>> pc.Dict(d).group_by_key(lambda k: k.split("_")[0])
        {'user': {'user_1': 10, 'user_2': 20}, 'admin': {'admin_1': 100}}

        ```
        """

        def _group_by_key(data: dict[K, V]) -> dict[G, dict[K, V]]:
            def _(kv: tuple[K, V]) -> G:
                return func(kv[0])

            return cz.dicttoolz.valmap(dict, cz.itertoolz.groupby(_, data.items()))

        return self._new(_group_by_key)

    def group_by_key_agg[G, R](
        self, key_func: Callable[[K], G], agg_func: Callable[[Dict[K, V]], R]
    ) -> Dict[G, R]:
        """Group by key function, then apply aggregation function to each sub-dict.

        Args:
            key_func(Callable[[K], G]): Function to determine the group for each key.
            agg_func(Callable[[Dict[K, V]], R]): Function to aggregate each sub-dictionary.

        Returns:
            Dict[G, R]: Grouped and aggregated Dict.

        This avoids materializing intermediate `dict` objects if you only need
        an aggregated result for each group.
        ```python
        >>> import pyochain as pc
        >>>
        >>> data = {"user_1": 10, "user_2": 20, "admin_1": 100}
        >>> pc.Dict(data).group_by_key_agg(
        ...     key_func=lambda k: k.split("_")[0],
        ...     agg_func=lambda d: d.iter_values().sum(),
        ... )
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
        ...         "count": sub_dict.iter_keys().length(),
        ...         "total_size": sub_dict.iter_values().sum(),
        ...         "max_size": sub_dict.iter_values().max(),
        ...         "files": sub_dict.iter_keys().sort().into(list),
        ...     }
        >>>
        >>> pc.Dict(data_files).group_by_key_agg(
        ...     key_func=lambda k: k.split(".")[-1], agg_func=get_stats
        ... ).iter_items().sort().into(pc.Dict.from_) # doctest: +NORMALIZE_WHITESPACE
        {'log': {'count': 2,
                'total_size': 25,
                'max_size': 20,
                'files': ['file_b.log', 'file_d.log']},
        'txt': {'count': 2,
                'total_size': 150,
                'max_size': 100,
                'files': ['file_a.txt', 'file_c.txt']}}

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
        """Group by value function, then apply aggregation function to each sub-dict.

        Args:
            value_func(Callable[[V], G]): Function to determine the group for each value.
            agg_func(Callable[[Dict[K, V]], R]): Function to aggregate each sub-dictionary.

        Returns:
            Dict[G, R]: Grouped and aggregated Dict.

        This avoids materializing intermediate `dict` objects if you only need
        an aggregated result for each group.
        ```python
        >>> import pyochain as pc
        >>>
        >>> data = {"math": "A", "physics": "B", "english": "A"}
        >>> pc.Dict(data).group_by_value_agg(
        ...     value_func=lambda grade: grade,
        ...     agg_func=lambda d: d.iter_keys().length(),
        ... )
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
        ... ).iter_items().sort().into(pc.Dict.from_)
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

    def flatten(
        self: Dict[str, Any],
        sep: str = ".",
        max_depth: int | None = None,
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary, concatenating keys with the specified separator.

        Args:
            sep (str): Separator to use when concatenating keys
            max_depth (int | None): Maximum depth to flatten. If None, flattens completely.

        Returns:
            Dict[str, Any]: Flattened Dict with concatenated keys.
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "config": {"params": {"retries": 3, "timeout": 30}, "mode": "fast"},
        ...     "version": 1.0,
        ... }
        >>> pc.Dict(data).flatten() # doctest: +NORMALIZE_WHITESPACE
        {'config.params.retries': 3,
        'config.params.timeout': 30,
        'config.mode': 'fast',
        'version': 1.0}
        >>>
        >>> pc.Dict(data).flatten(sep="_") # doctest: +NORMALIZE_WHITESPACE
        {'config_params_retries': 3,
        'config_params_timeout': 30,
        'config_mode': 'fast',
        'version': 1.0}
        >>>
        >>> pc.Dict(data).flatten(max_depth=1) # doctest: +NORMALIZE_WHITESPACE
        {'config.params': {'retries': 3, 'timeout': 30},
        'config.mode': 'fast',
        'version': 1.0}

        ```
        """

        def _flatten(
            d: Mapping[Any, Any],
            parent_key: str = "",
            current_depth: int = 1,
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

    def unpivot(self: Dict[str, Mapping[str, Any]]) -> Dict[str, dict[str, Any]]:
        """Unpivot a nested dictionary by swapping rows and columns.

        Returns:
            Dict[str, dict[str, Any]]: Unpivoted Dict with columns as keys and rows as sub-dicts.

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
        """Set a nested key path and return a new Dict with new, potentially nested, key value pair.

        Args:
            *keys (K): keys representing the nested path.
            value (V): Value to set at the specified nested path.

        Returns:
            Dict[K, V]: Dict with the new nested key-value pair.
        ```python
        >>> import pyochain as pc
        >>> purchase = {
        ...     "name": "Alice",
        ...     "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        ...     "credit card": "5555-1234-1234-1234",
        ... }
        >>> pc.Dict(purchase).with_nested_key(
        ...     "order", "costs", value=[0.25, 1.00]
        ... ) # doctest: +NORMALIZE_WHITESPACE
        {'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.0]},
        'credit card': '5555-1234-1234-1234'}

        ```
        """

        def _with_nested_key(data: dict[K, V]) -> dict[K, V]:
            return cz.dicttoolz.assoc_in(data, keys, value=value)

        return self._new(_with_nested_key)

    def pluck[U: str | int](self: Dict[U, Any], *keys: str) -> Dict[U, Any]:
        """Extract values from nested dictionaries using a sequence of keys.

        Args:
            *keys (str): keys to extract values from the nested dictionaries.

        Returns:
            Dict[U, Any]: Dict with extracted values from nested dictionaries.
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "person1": {"name": "Alice", "age": 30},
        ...     "person2": {"name": "Bob", "age": 25},
        ... }
        >>> pc.Dict(data).pluck("name")
        {'person1': 'Alice', 'person2': 'Bob'}

        ```
        """
        getter = partial(cz.dicttoolz.get_in, keys)

        def _pluck(data: Mapping[U, Any]) -> dict[U, Any]:
            return cz.dicttoolz.valmap(getter, data)

        return self._new(_pluck)

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

    def drop_nones(self, *, remove_empty: bool = True) -> Dict[K, V]:
        """Recursively drop None values from the dictionary.

        Options to also remove empty dicts and lists.

        Args:
            remove_empty (bool): If True (default), removes `None`, `{}` and `[]`.

        Returns:
            Dict[K, V]: Dict with None values and optionally empty structures removed.

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
        >>> p_data.drop_nones()
        {'a': 1, 'e': {'g': 2}, 'h': [1], 'i': 0}
        >>>
        >>> p_data.drop_nones()
        {'a': 1, 'e': {'g': 2}, 'h': [1], 'i': 0}
        >>>
        >>> p_data.drop_nones(remove_empty=False) # doctest: +NORMALIZE_WHITESPACE
        {'a': 1,
        'b': None,
        'c': {},
        'd': [],
        'e': {'f': None, 'g': 2},
        'h': [1, None, {}],
        'i': 0}

        ```
        """

        def _drop_nones(
            data: dict[Any, Any] | list[Any],
        ) -> dict[Any, Any] | list[Any] | None:
            match data:
                case dict():
                    pruned_dict: dict[Any, Any] = {}
                    for k, v in data.items():
                        pruned_v = _drop_nones(v)

                        is_empty = remove_empty and (
                            pruned_v is None or pruned_v in ({}, [])
                        )
                        if not is_empty:
                            pruned_dict[k] = pruned_v
                    return pruned_dict if pruned_dict or not remove_empty else None

                case list():
                    pruned_list = [_drop_nones(item) for item in data]
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

        def _apply_drop_nones(data: dict[K, V]) -> dict[Any, Any]:
            result = _drop_nones(data)
            return result if isinstance(result, dict) else {}

        return self._new(_apply_drop_nones)
