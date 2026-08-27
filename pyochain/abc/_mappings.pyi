from abc import abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Generic, TypeVar, overload, override

from _typeshed import SupportsGetItem, SupportsKeysAndGetItem

from pyochain import Option, Result
from pyochain.abc import (
    PyoCollection,
    PyoItemsView,
    PyoKeysView,
    PyoValuesView,
)

# NOTE: We are forced to use legacy `TypeVar` syntax here due to concrete limitations of the typing system. Typeshed explicitely ignore some typing warnings.
# https://github.com/python/typing/pull/273
_K = TypeVar("_K")
_V_co = TypeVar("_V_co", covariant=True)

# pyrefly: ignore [implicit-abstract-class]
class PyoMapping(PyoCollection[_K], Mapping[_K, _V_co], Generic[_K, _V_co]):  # pyright: ignore[reportImplicitAbstractClass]  # ruff:ignore[non-pep695-generic-class]
    """Extends `PyoCollection[K]` and `collections.abc.Mapping[K, V]`.

    Serves as a base class for pyochain mappings, such as `Dict`.

    Any concrete subclass must implement the required `Mapping` dunder methods:

    - `__getitem__`
    - `__iter__`
    - `__len__`
    """

    @abstractmethod
    @override
    def __getitem__(self, key: _K, /) -> _V_co: ...
    @override
    def __contains__(self, key: object, /) -> bool: ...
    @override
    def __eq__(self, other: object, /) -> bool: ...
    @override
    def keys(self) -> PyoKeysView[_K]:
        """Return a view of the `Mapping` keys.

        Returns:
            PyoKeysView[_K]: A view of the dictionary's keys.

        Example:
            ```python
            from pyochain import Dict
            from pyochain.abc import PyoKeysView

            data = Dict(a=1, b=2)
            keys = data.keys()
            assert isinstance(keys, PyoKeysView)
            assert keys.pipe(tuple) == ("a", "b")
            assert repr(keys) == "PyoKeysView(Dict('a': 1, 'b': 2))"
            ```
        """

    @override
    def values(self) -> PyoValuesView[_V_co]:
        """Return a view of the `Mapping` values.

        Returns:
            PyoValuesView[_V_co]: A view of the dictionary's values.

        Example:
            ```python
            from pyochain import Dict
            from pyochain.abc import PyoValuesView

            data = Dict(a=1, b=2)
            values = data.values()
            assert isinstance(values, PyoValuesView)
            assert values.pipe(tuple) == (1, 2)
            assert repr(values) == "PyoValuesView(Dict('a': 1, 'b': 2))"
            ```
        """

    @override
    def items(self) -> PyoItemsView[_K, _V_co]:
        """Return a view of the `Mapping` items.

        Returns:
            PyoItemsView[_K, _V_co]: A view of the dictionary's (key, value) pairs.

        Example:
            ```python
            from pyochain import Dict
            from pyochain.abc import PyoItemsView

            data = Dict(a=1, b=2)
            items = data.items()
            assert isinstance(items, PyoItemsView)
            assert items.pipe(tuple) == (("a", 1), ("b", 2))
            assert repr(items) == "PyoItemsView(Dict('a': 1, 'b': 2))"
            ```
        """

    @overload
    def get(self, key: _K, /) -> _V_co | None: ...
    @overload
    # pyrefly: ignore [invalid-variance]
    def get(
        self,
        key: _K,
        default: _V_co,  # pyright: ignore[reportGeneralTypeIssues]
        /,
    ) -> _V_co: ...
    @overload
    def get[T](self, key: _K, default: T, /) -> _V_co | T: ...
    @override
    def get[T](self, key: _K, default: T | None = None, /) -> _V_co | T | None: ...
    def get_item(self, key: _K) -> Option[_V_co]:
        """Retrieve a value from the `MutableMapping`.

        Returns `Some(value)` if the **key** exists, or `Null` if it does not.

        Args:
            key (_K): The key to look up.

        Returns:
            Option[_V_co]: `Some(value)` that is associated with the **key**, or `Null` if not found.

        Example:
            ```python
            from pyochain import Dict, Some

            data = Dict(a=1)
            item = data.get_item("a")
            assert item == Some(1)
            assert data.get_item("x").unwrap_or("Not Found") == "Not Found"
            ```
        """

class PyoMutableMapping[K, V](PyoMapping[K, V], MutableMapping[K, V]):  # pyright: ignore[reportImplicitAbstractClass]
    """Extends `PyoMapping[K, V]` and `collections.abc.MutableMapping[K, V]`.

    Serves as a base class for pyochain mutable mappings, such as `Dict`.

    Any concrete subclass must implement the required `MutableMapping` dunder methods:

    - `__getitem__`
    - `__setitem__`
    - `__delitem__`
    - `__iter__`
    - `__len__`

    """

    @abstractmethod
    @override
    def __setitem__(self, key: K, value: V, /) -> None: ...
    @abstractmethod
    @override
    def __delitem__(self, key: K, /) -> None: ...
    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    @override
    def pop[T](self, key: K, default: T = ..., /) -> V | T: ...
    @override
    def popitem(self) -> tuple[K, V]:
        """Remove and return a (key, value) pair from the `PyoMutableMapping`.

        Pairs are returned in LIFO order in the default implementation.

        It can be useful to destructively iterate over *self*, as often used in set algorithms.

        Raises `KeyError` if the `PyoMutableMapping` is empty.

        Returns:
            tuple[K, V]: The removed (key, value) pair.

        Example:
            ```python
            from pyochain import Dict, Ok, Err

            d = Dict(a=1, b=2)
            popped = d.popitem()
            assert popped == ("b", 2)
            popped = d.popitem()
            assert popped == ("a", 1)
            try:
                res = Ok(d.popitem())
            except KeyError as e:
                res = Err(e)
            assert repr(res) == '''Err(KeyError('popitem(): dictionary is empty'))'''
            # LIFO order is preserved
            d = Dict(a=1, b=2, c=3)
            assert d.popitem() == ("c", 3)
            assert d.popitem() == ("b", 2)
            assert d.popitem() == ("a", 1)
            assert d.is_empty()
            ```
        """
    @override
    def clear(self) -> None: ...
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /) -> None: ...
    @overload
    def update(
        self: SupportsGetItem[str, V], m: SupportsKeysAndGetItem[str, V], /, **kwargs: V
    ) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[K, V]], /) -> None: ...
    @overload
    def update(
        self: SupportsGetItem[str, V], m: Iterable[tuple[str, V]], /, **kwargs: V
    ) -> None: ...
    @overload
    def update(self: SupportsGetItem[str, V], /, **kwargs: V) -> None: ...
    @override
    def update(self, m: object = ..., /, **kwargs: V) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """In-place update of *self*, with key-value pairs from another `Mapping` or `Iterable`.

        Keywords will be applied after the *m* argument, allowing for additional updates.

        If a key already exists in *self*, its value will be overwritten.

        If a given key is present in both *m* and *kwargs*, the value from the latter will be the one effectively present in the resulting `Dict`.

        Args:
            m (object): The `Mapping` or `Iterable` to update *self* from.
            **kwargs (V): Additional key-value pairs to update *self* with.

        Example:
            ```python
            from pyochain import Dict

            d = Dict({1: "a", 2: "b"})
            d.update({2: "c", 3: "d"})
            assert d == Dict({1: "a", 2: "c", 3: "d"})
            d.update([(3, "e"), (4, "f")])
            assert d == Dict({1: "a", 2: "c", 3: "e", 4: "f"})
            d1 = Dict({"John": 30, "Jane": 25})
            d1.update({"John": 26, "Doe": 22}, John=31, Mary=28)
            assert d1 == Dict({"John": 31, "Jane": 25, "Doe": 22, "Mary": 28})
            ```
        """
    @overload
    def setdefault[T](
        self: MutableMapping[K, T | None], key: K, default: None = None, /
    ) -> T | None: ...
    @overload
    def setdefault(self, key: K, default: V, /) -> V: ...
    @override
    def setdefault[T](self, key: K, default: object = None, /) -> V: ...
    """Return the value for `key` if it is in *self*, else insert `key` with a value of `default` into *self* and return `default`.

        Args:
            key (K): The key to look for in *self*.
            default (object): The value to insert if `key` is not found. Defaults to `None`.

        Returns:
            V | T | None: The value associated with `key`, or `default` if `key` was not found.

        Example:
            ```python
            from pyochain import Dict
            d = Dict(a=1, b=2)
            assert d.setdefault("b", 3) == 2
            assert d.setdefault("c", 4) == 4
            assert d == Dict(a=1, b=2, c=4)
            assert d.setdefault("d") is None
            assert d == Dict(a=1, b=2, c=4, d=None)

            ```
        """
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
            from pyochain import Dict, Some

            data = Dict(())
            assert data.insert(37, "a").is_none()
            assert not data.is_empty()

            assert data.insert(37, "b") == Some("a")
            assert data.insert(37, "c") == Some("b")
            assert data[37] == "c"
            ```
        """
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
            from pyochain import Dict, Err

            d = Dict(())
            assert d.try_insert("a", 1).unwrap() == 1

            x = d.try_insert("a", 2)
            assert x.is_err()
            unwrapped = x.unwrap_err()
            assert isinstance(unwrapped, KeyError)
            assert repr(unwrapped) == "KeyError('Key a already exists with value 1.')"

            assert d == Dict(a=1)
            ```
        """

    def remove(self, key: K) -> Option[V]:
        """Remove a **key** from the `MutableMapping` and return its value if it existed.

        Equivalent to `dict.pop(key, None)`, with an `Option` return type.

        Args:
            key (K): The key to remove.

        Returns:
            Option[V]: The value associated with the removed **key**, or `None` if the **key** was not present.

        Example:
            ```python
            from pyochain import Dict, Some

            data = Dict(a=1, b=2)
            assert data.remove("a") == Some(1)
            assert data.remove("c").is_none()
            ```
        """

    def remove_entry(self, key: K) -> Option[tuple[K, V]]:
        """Remove a key from the `MutableMapping` and return the item if it existed.

        Return an `Option[tuple[K, V]]` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[tuple[K, V]]: `Some((key, value))` pair associated with the removed key, or `None` if the **key** was not present.

        Example:
            ```python
            from pyochain import Dict, Some

            data = Dict(a=1, b=2)
            assert data.remove_entry("a") == Some(("a", 1))
            assert data.remove_entry("c").is_none()
            ```
        """
