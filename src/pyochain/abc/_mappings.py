from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, MappingView, MutableMapping
from typing import TYPE_CHECKING, override

from ..rs import NONE, Err, Ok, Option, Result, Some, option
from ._collection import PyoCollection

if TYPE_CHECKING:
    from .._set import PyoItemsView, PyoKeysView, PyoValuesView
    from ..rs import Option, Result


class PyoMappingView[T](MappingView, PyoCollection[T], ABC):
    """Extends both `MappingView` from `collections.abc` and `PyoCollection[T]`.

    Is the base class shared by the views returned by `PyoMapping` methods.

    Any concrete subclass must implement the required `MappingView` dunder methods:

    - `__contains__`
    - `__iter__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoMapping[K, V](PyoCollection[K], Mapping[K, V], ABC):
    """Extends `PyoCollection[K]` and `collections.abc.Mapping[K, V]`.

    Serves as a base class for pyochain mappings, such as `Dict`.

    Any concrete subclass must implement the required `Mapping` dunder methods:

    - `__getitem__`
    - `__iter__`
    - `__len__`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def all_unique(self) -> bool:
        return True

    @override
    def keys(self) -> PyoKeysView[K]:
        """Return a view of the `Mapping` keys.

        Returns:
            PyoKeysView[K]: A view of the dictionary's keys.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.keys()
            PyoKeysView(Dict(1: 'a', 2: 'b'))

            ```
        """
        from .._set import PyoKeysView

        return PyoKeysView(self)

    @override
    def values(self) -> PyoValuesView[V]:
        """Return a view of the `Mapping` values.

        Returns:
            PyoValuesView[V]: A view of the dictionary's values.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.values()
            PyoValuesView(Dict(1: 'a', 2: 'b'))

            ```
        """
        from .._set import PyoValuesView

        return PyoValuesView(self)

    @override
    def items(self) -> PyoItemsView[K, V]:
        """Return a view of the `Mapping` items.

        Returns:
            PyoItemsView[K, V]: A view of the dictionary's (key, value) pairs.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.items()
            PyoItemsView(Dict(1: 'a', 2: 'b'))

            ```
        """
        from .._set import PyoItemsView

        return PyoItemsView(self)


class PyoMutableMapping[K, V](PyoMapping[K, V], MutableMapping[K, V], ABC):
    """Extends `PyoMapping[K, V]` and `collections.abc.MutableMapping[K, V]`.

    Serves as a base class for pyochain mutable mappings, such as `Dict`.

    Any concrete subclass must implement the required `MutableMapping` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__iter__`
    - `__len__`

    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def insert(self, key: K, value: V) -> Option[V]:
        """Insert a key-value pair into the `MutableMapping`.

        If the `MutableMapping` did not have this **key** present, `NONE` is returned.

        If the `MutableMapping` did have this **key** present, the **value** is updated, and the old value is returned.

        The **key** is not updated.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Option[V]: The previous value associated with the key, or None if the key was not present.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict(())
            >>> data.insert(37, "a")
            NONE
            >>> data.is_empty()
            False

            >>> data.insert(37, "b")
            Some('a')
            >>> data.insert(37, "c")
            Some('b')
            >>> data[37]
            'c'

            ```
        """
        previous = self.get(key, None)
        self[key] = value
        return option(previous)

    def try_insert(self, key: K, value: V) -> Result[V, KeyError]:
        """Tries to insert a key-value pair into the `MutableMapping`, and returns a `Result[V, KeyError]` containing the value in the entry (if successful).

        If the `MutableMapping` already had this **key** present, nothing is updated, and an error containing the occupied entry and the value is returned.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Result[V, KeyError]: `Ok` containing the value if the **key** was not present, or `Err` containing a `KeyError` if the **key** already existed.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d = Dict(())
            >>> d.try_insert(37, "a").unwrap()
            'a'
            >>> d.try_insert(37, "b")
            Err(KeyError('Key 37 already exists with value a.'))

            ```
        """
        if key in self:
            return Err(KeyError(f"Key {key} already exists with value {self[key]}."))
        self[key] = value
        return Ok(value)

    def remove(self, key: K) -> Option[V]:
        """Remove a **key** from the `MutableMapping` and return its value if it existed.

        Equivalent to `dict.pop(key, None)`, with an `Option` return type.

        Args:
            key (K): The key to remove.

        Returns:
            Option[V]: The value associated with the removed **key**, or `None` if the **key** was not present.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.remove(1)
            Some('a')
            >>> data.remove(3)
            NONE

            ```
        """
        return option(self.pop(key, None))

    def remove_entry(self, key: K) -> Option[tuple[K, V]]:
        """Remove a key from the `MutableMapping` and return the item if it existed.

        Return an `Option[tuple[K, V]]` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[tuple[K, V]]: `Some((key, value))` pair associated with the removed key, or `None` if the **key** was not present.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.remove_entry(1)
            Some((1, 'a'))
            >>> data.remove_entry(3)
            NONE

            ```
        """
        return Some((key, self.pop(key))) if key in self else NONE

    def get_item(self, key: K) -> Option[V]:
        """Retrieve a value from the `MutableMapping`.

        Returns `Some(value)` if the **key** exists, or `None` if it does not.

        Args:
            key (K): The key to look up.

        Returns:
            Option[V]: `Some(value)` that is associated with the **key**, or `None` if not found.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict.from_ref({"a": 1})
            >>> data.get_item("a")
            Some(1)
            >>> data.get_item("x").unwrap_or("Not Found")
            'Not Found'

            ```
        """
        return option(self.get(key, None))
