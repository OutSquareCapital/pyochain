from abc import abstractmethod
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MappingView,
    MutableMapping,
    Sized,
    ValuesView,
)
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
    override,
)

from _typeshed import (
    SupportsGetItem,
    SupportsGetItemViewable,
    SupportsKeysAndGetItem,
    Viewable,
)

from pyochain import Option, Result, SetMut
from pyochain.abc import PyoCollection, PyoSet, PyoSized

class PyoMappingView(MappingView, PyoSized):
    """Extends both `MappingView` from `collections.abc` and `PyoCollection[T]`.

    Is the base class shared by the views returned by `PyoMapping` methods.
    """

    _mapping: Sized
    def __init__(self, mapping: Sized) -> None: ...
    @override
    def __len__(self) -> int: ...

# NOTE: We are forced to use legacy `TypeVar` syntax here due to concrete limitations of the typing system. Typeshed explicitely ignore some typing warnings.
# https://github.com/python/typing/pull/273
_K = TypeVar("_K")
_K_co = TypeVar("_K_co", covariant=True)
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
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.keys()
            PyoKeysView(Dict(1: 'a', 2: 'b'))

            ```
        """

    @override
    def values(self) -> PyoValuesView[_V_co]:
        """Return a view of the `Mapping` values.

        Returns:
            PyoValuesView[_V_co]: A view of the dictionary's values.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.values()
            PyoValuesView(Dict(1: 'a', 2: 'b'))

            ```
        """

    @override
    def items(self) -> PyoItemsView[_K, _V_co]:
        """Return a view of the `Mapping` items.

        Returns:
            PyoItemsView[_K, _V_co]: A view of the dictionary's (key, value) pairs.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> data = Dict({1: "a", 2: "b"})
            >>> data.items()
            PyoItemsView(Dict(1: 'a', 2: 'b'))

            ```
        """

    @overload
    def get(self, key: _K, /) -> _V_co | None: ...
    @overload
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

class PyoKeysView(PyoMappingView, PyoSet[_K_co], KeysView[_K_co]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the keys in a pyochain mapping.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    def __init__(self, mapping: Viewable[_K_co]) -> None: ...
    @classmethod
    @override
    def _from_iterable[S](cls, it: Iterable[S], /) -> set[S]: ...
    @override
    def __contains__(self, key: object, /) -> bool: ...
    @override
    def __iter__(self) -> Iterator[_K_co]: ...
    @override
    def __and__(self, other: Iterable[Any], /) -> SetMut[_K_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rand__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __or__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ror__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __sub__[T](self, other: Iterable[Any], /) -> SetMut[_K_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rsub__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __xor__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rxor__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[_K_co]: ...
    @override
    def union[S, T](self: PyoKeysView[S], other: Iterable[T]) -> SetMut[S | T]: ...
    @override
    def difference(self, other: Iterable[Any]) -> SetMut[_K_co]: ...
    @override
    def symmetric_difference[S, T](
        self: PyoKeysView[S], other: Iterable[T]
    ) -> SetMut[S | T]: ...

class PyoValuesView[V](PyoMappingView, PyoCollection[V], ValuesView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """
    def __init__(self, mapping: SupportsGetItemViewable[Any, V]) -> None: ...
    @override
    def __contains__(self, value: object, /) -> bool: ...
    @override
    def __iter__(self) -> Iterator[V]: ...

class PyoItemsView(  # pyright: ignore[reportUnsafeMultipleInheritance]
    PyoMappingView,
    PyoSet[tuple[_K_co, _V_co]],
    ItemsView[_K_co, _V_co],
    Generic[_K_co, _V_co],  # ruff:ignore[non-pep695-generic-class]
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping::items`: Method that returns this view.
    """
    @classmethod
    @override
    def _from_iterable[S](cls, it: Iterable[S], /) -> set[S]: ...
    @override
    # pyrefly: ignore [bad-override]
    def __contains__(self, item: tuple[object, object], /) -> bool: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __iter__(self) -> Iterator[tuple[_K_co, _V_co]]: ...
    @override
    def __and__(self, other: Iterable[Any], /) -> SetMut[tuple[_K_co, _V_co]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rand__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __or__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ror__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __sub__[T](self, other: Iterable[Any], /) -> SetMut[tuple[_K_co, _V_co]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rsub__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __xor__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rxor__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[tuple[_K_co, _V_co]]: ...
    @override
    def union[T](self, other: Iterable[T]) -> SetMut[tuple[_K_co, _V_co] | T]: ...
    @override
    def difference(self, other: Iterable[Any]) -> SetMut[tuple[_K_co, _V_co]]: ...
    @override
    def symmetric_difference[T](
        self, other: Iterable[T]
    ) -> SetMut[tuple[_K_co, _V_co] | T]: ...

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
            >>> from pyochain import Dict, Ok, Err
            >>> d = Dict({1: "a", 2: "b"})
            >>> d.popitem()
            (2, 'b')
            >>> d.popitem()
            (1, 'a')
            >>> try:
            ...     res = Ok(d.popitem())
            ... except KeyError as e:
            ...     res = Err(e)
            >>> res
            Err(KeyError('popitem(): dictionary is empty'))
            >>> # LIFO order is preserved
            >>> d = Dict({1: "a", 2: "b", 3: "c"})
            >>> d.popitem()
            (3, 'c')
            >>> d.popitem()
            (2, 'b')
            >>> d.popitem()
            (1, 'a')

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

        Note:
        If a given key is present in both *m* and *kwargs*, the value from the latter will be the one effectively present in the resulting `Dict`.

        Args:
            m (SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] | None): The `Mapping` or `Iterable` to update *self* from.
            **kwargs (V): Additional key-value pairs to update *self* with.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d = Dict({1: "a", 2: "b"})
            >>> d.update({2: "c", 3: "d"})
            >>> d
            Dict(1: 'a', 2: 'c', 3: 'd')
            >>> d.update([(3, "e"), (4, "f")])
            >>> d
            Dict(1: 'a', 2: 'c', 3: 'e', 4: 'f')
            >>> d1 = Dict({"John": 30, "Jane": 25})
            >>> d1.update({"John": 26, "Doe": 22}, John=31, Mary=28)
            >>> d1
            Dict('John': 31, 'Jane': 25, 'Doe': 22, 'Mary': 28)

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
            >>> from pyochain import Dict
            >>> d = Dict({1: "a", 2: "b"})
            >>> d.setdefault(2, "c")
            'b'
            >>> d.setdefault(3, "d")
            'd'
            >>> d
            Dict(1: 'a', 2: 'b', 3: 'd')
            >>> d.setdefault(4)
            None
            >>> d
            Dict(1: 'a', 2: 'b', 3: 'd', 4: None)

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
