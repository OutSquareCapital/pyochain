from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeGuard

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from ._main import LazyDict


class FilterDict[K, V](MappingWrapper[K, V]):
    def filter_keys(self, predicate: Callable[[K], bool]) -> LazyDict[K, V]:
        """
        Return keys that satisfy predicate.

        Args:
            predicate: Function to determine if a key should be included.
        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
        >>> pc.Dict(d).filter_keys(lambda x: x % 2 == 0).unwrap()
        {2: 3, 4: 5}

        ```
        """
        return self._new(partial(cz.dicttoolz.keyfilter, predicate))

    def filter_values(self, predicate: Callable[[V], bool]) -> LazyDict[K, V]:
        """
        Return items whose values satisfy predicate.

        Args:
            predicate: Function to determine if a value should be included.
        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
        >>> pc.Dict(d).filter_values(lambda x: x % 2 == 0).unwrap()
        {1: 2, 3: 4}
        >>> pc.Dict(d).filter_values(lambda x: not x > 3).unwrap()
        {1: 2, 2: 3}

        ```
        """
        return self._new(partial(cz.dicttoolz.valfilter, predicate))

    def filter_items(self, predicate: Callable[[tuple[K, V]], bool]) -> LazyDict[K, V]:
        """
        Filter items by predicate applied to (key, value) tuples.

        Args:
            predicate: Function to determine if a (key, value) pair should be included.
        Example:
        ```python
        >>> import pyochain as pc
        >>> def isvalid(item):
        ...     k, v = item
        ...     return k % 2 == 0 and v < 4
        >>> d = pc.Dict({1: 2, 2: 3, 3: 4, 4: 5})
        >>>
        >>> d.filter_items(isvalid).unwrap()
        {2: 3}
        >>> d.filter_items(lambda kv: not isvalid(kv)).unwrap()
        {1: 2, 3: 4, 4: 5}

        ```
        """
        return self._new(partial(cz.dicttoolz.itemfilter, predicate))

    def filter_kv(self, predicate: Callable[[K, V], bool]) -> LazyDict[K, V]:
        """
        Filter items by predicate applied to unpacked (key, value) tuples.

        Args:
            predicate: Function to determine if a key-value pair should be included.
        Example:
        ```python
        >>> import pyochain as pc
        >>> def isvalid(key, value):
        ...     return key % 2 == 0 and value < 4
        >>> d = pc.Dict({1: 2, 2: 3, 3: 4, 4: 5})
        >>>
        >>> d.filter_kv(isvalid).unwrap()
        {2: 3}
        >>> d.filter_kv(lambda k, v: not isvalid(k, v)).unwrap()
        {1: 2, 3: 4, 4: 5}

        ```
        """

        def _filter_kv(data: dict[K, V]) -> dict[K, V]:
            def _(kv: tuple[K, V]) -> bool:
                return predicate(kv[0], kv[1])

            return cz.dicttoolz.itemfilter(_, data)

        return self._new(_filter_kv)

    def filter_attr[U](self, attr: str, dtype: type[U] = object) -> LazyDict[K, U]:
        """
        Filter values that have a given attribute.

        This does not enforce type checking at runtime for performance considerations.

        Args:
            attr: Attribute name to check for.
            dtype: Optional expected type of the attribute for type hinting.
        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"a": "hello", "b": "world", "c": 2, "d": 5}).filter_attr(
        ...     "capitalize", str
        ... ).unwrap()
        {'a': 'hello', 'b': 'world'}

        ```
        """

        def _filter_attr(data: dict[K, V]) -> dict[K, U]:
            def has_attr(x: V) -> TypeGuard[U]:
                return hasattr(x, attr)

            return cz.dicttoolz.valfilter(has_attr, data)

        return self._new(_filter_attr)

    def filter_type[R](self, typ: type[R]) -> LazyDict[K, R]:
        """
        Filter values by type.

        Args:
            typ: Type to filter values by.
        Example:
        ```python
        >>> import pyochain as pc
        >>> data = {"a": "one", "b": "two", "c": 3, "d": 4}
        >>> pc.Dict(data).filter_type(str).unwrap()
        {'a': 'one', 'b': 'two'}

        ```
        """

        def _filter_type(data: dict[K, V]) -> dict[K, R]:
            def _(x: V) -> TypeGuard[R]:
                return isinstance(x, typ)

            return cz.dicttoolz.valfilter(_, data)

        return self._new(_filter_type)

    def filter_callable(self) -> LazyDict[K, Callable[..., Any]]:
        """
        Filter values that are callable.
        ```python
        >>> import pyochain as pc
        >>> def foo():
        ...     pass
        >>> data = {1: "one", 2: "two", 3: foo, 4: print}
        >>> pc.Dict(data).filter_callable().map_values(lambda x: x.__name__).unwrap()
        {3: 'foo', 4: 'print'}

        ```
        """

        def _filter_callable(data: dict[K, V]) -> dict[K, Callable[..., Any]]:
            def _(x: V) -> TypeGuard[Callable[..., Any]]:
                return callable(x)

            return cz.dicttoolz.valfilter(_, data)

        return self._new(_filter_callable)

    def filter_subclass[U: type[Any], R](
        self: FilterDict[K, U], parent: type[R], keep_parent: bool = True
    ) -> LazyDict[K, type[R]]:
        """
        Filter values that are subclasses of a given parent class.

        Args:
            parent: Parent class to check against.
            keep_parent: Whether to include the parent class itself. Defaults to True.

        ```python
        >>> import pyochain as pc
        >>> class A:
        ...     pass
        >>> class B(A):
        ...     pass
        >>> class C:
        ...     pass
        >>> def name(cls: type[Any]) -> str:
        ...     return cls.__name__
        >>> data = pc.Dict({"first": A, "second": B, "third": C})
        >>> data.filter_subclass(A).map_values(name).unwrap()
        {'first': 'A', 'second': 'B'}
        >>> data.filter_subclass(A, keep_parent=False).map_values(name).unwrap()
        {'second': 'B'}

        ```
        """

        def _filter_subclass(data: dict[K, U]) -> dict[K, type[R]]:
            def _(x: type[Any]) -> TypeGuard[type[R]]:
                if keep_parent:
                    return issubclass(x, parent)
                else:
                    return issubclass(x, parent) and x is not parent

            return cz.dicttoolz.valfilter(_, data)

        return self._new(_filter_subclass)

    def intersect_keys(self, *others: Mapping[K, V]) -> LazyDict[K, V]:
        """
        Keep only keys present in self and all others mappings.

        Args:
            *others: Other mappings to intersect keys with.

        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"b": 10, "c": 20}
        >>> d3 = {"c": 30}
        >>> pc.Dict(d1).intersect_keys(d2, d3).unwrap()
        {'c': 3}

        ```
        """

        def _intersect_keys(data: dict[K, V]) -> dict[K, V]:
            self_keys = set(data.keys())
            for other in others:
                self_keys.intersection_update(other.keys())
            return {k: data[k] for k in self_keys}

        return self._new(_intersect_keys)

    def diff_keys(self, *others: Mapping[K, V]) -> LazyDict[K, V]:
        """
        Keep only keys present in self but not in others mappings.

        Args:
            *others: Other mappings to exclude keys from.

        ```python
        >>> import pyochain as pc
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"b": 10, "d": 40}
        >>> d3 = {"c": 30}
        >>> pc.Dict(d1).diff_keys(d2, d3).unwrap()
        {'a': 1}

        ```
        """

        def _diff_keys(data: dict[K, V]) -> dict[K, V]:
            self_keys = set(data.keys())
            for other in others:
                self_keys.difference_update(other.keys())
            return {k: data[k] for k in self_keys}

        return self._new(_diff_keys)
