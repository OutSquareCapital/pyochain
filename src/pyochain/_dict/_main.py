from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._core import SupportsKeysAndGetItem
from ._filters import FilterDict
from ._groups import GroupsDict
from ._iter import IterDict
from ._joins import JoinsDict
from ._maps import MapDict
from ._nested import NestedDict
from ._process import ProcessDict


class DictCommonMethods[K, V](
    ProcessDict[K, V],
    IterDict[K, V],
    NestedDict[K, V],
    MapDict[K, V],
    JoinsDict[K, V],
    FilterDict[K, V],
    GroupsDict[K, V],
):
    pass


class Dict[K, V](DictCommonMethods[K, V]):
    """
    Wrapper for Python dictionaries with chainable methods.
    """

    __slots__ = ()

    @staticmethod
    def from_[G, I](
        data: Mapping[G, I] | Iterable[tuple[G, I]] | SupportsKeysAndGetItem[G, I],
    ) -> Dict[G, I]:
        """
        Create a Dict from a convertible value.

        Args:
            data: A mapping, Iterable of tuples, or object supporting keys and item access to convert into a Dict.

        Returns:
            A Dict instance containing the data from the input.

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
        >>> pc.Dict.from_(MyMapping()).inner()
        {1: 'a', 2: 'b', 3: 'c'}
        >>> pc.Dict.from_([("d", "e"), ("f", "g")]).inner()
        {'d': 'e', 'f': 'g'}

        ```
        """
        return Dict(dict(data))

    @staticmethod
    def from_object(obj: object) -> Dict[str, Any]:
        """
        Create a Dict from an object's __dict__ attribute.

        Args:
            obj: The object whose `__dict__` attribute will be used to create the Dict.

        ```python
        >>> import pyochain as pc
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Alice", 30)
        >>> pc.Dict.from_object(person).inner()
        {'name': 'Alice', 'age': 30}

        ```
        """
        return Dict(obj.__dict__)

    def pivot(self, *indices: int) -> Dict[Any, Any]:
        """
        Pivot a nested dictionary by rearranging the key levels according to order.

        Syntactic sugar for to_arrays().rearrange(*indices).to_records()

        Args:
            indices: Indices specifying the new order of key levels

        Returns:
            Pivoted dictionary with keys rearranged

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {"A": {"X": 1, "Y": 2}, "B": {"X": 3, "Y": 4}}
        >>> pc.Dict(d).pivot(1, 0).inner()
        {'X': {'A': 1, 'B': 3}, 'Y': {'A': 2, 'B': 4}}
        """

        return self.to_arrays().rearrange(*indices).to_records()
