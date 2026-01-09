from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Self, override

from .traits import PyoIterable

if TYPE_CHECKING:
    from ._iter import Iter
    from ._option import Option
    from ._result import Result
    from ._types import SupportsKeysAndGetItem


class Dict[K, V](PyoIterable[dict[K, V], K], MutableMapping[K, V]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    Accept the same input types as the built-in `dict`.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.

    Note:
        Prefer using `Dict.from_ref` when wrapping existing dictionaries to avoid unnecessary copying.

    Args:
        data (Mapping[K, V] | Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]): Initial data for the Dict.
    """

    _inner: dict[K, V]

    def __init__(
        self, data: Mapping[K, V] | Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]
    ) -> None:
        self._inner = dict(data)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, key: K) -> V:
        return self._inner[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._inner[key] = value

    def __delitem__(self, key: K) -> None:
        del self._inner[key]

    @staticmethod
    def from_ref[K1, V1](data: dict[K1, V1]) -> Dict[K1, V1]:
        """Wrap an existing `dict` without copying.

        This is the recommended way to create a `Dict` from foreign functions that return a standard Python `dict`.

        **Warning** ⚠️:
            Any modifications made to this `Dict` will also affect the original, and vice versa.

        Args:
            data (dict[K1, V1]): The dictionary to wrap.

        Returns:
            Dict[K1, V1]: A new `Dict` instance wrapping the provided dictionary.

        Example:
        ```python
        >>> import pyochain as pc
        >>> original_dict = {1: "a", 2: "b", 3: "c"}
        >>> dict_obj = pc.Dict.from_ref(original_dict)
        >>> dict_obj
        Dict(1: 'a', 2: 'b', 3: 'c')
        >>> dict_obj[1] = "z"
        >>> original_dict
        {1: 'z', 2: 'b', 3: 'c'}

        ```
        """
        instance: Dict[K1, V1] = Dict.__new__(Dict)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    def contains_key(self, key: K) -> bool:
        """Check if the `Dict` contains the specified key.

        This is equivalent to using the `in` keyword directly on the `Dict`.

        Args:
            key (K): The key to check for existence.

        Returns:
            bool: True if the key exists in the Dict, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.contains_key(1)
        True
        >>> data.contains_key(3)
        False

        ```
        """
        return key in self._inner

    @classmethod
    def new(cls) -> Self:
        """Create an empty `Dict`.

        Be sure to specify the key and value types when using this method, otherwise they will be unknown.

        Returns:
            Self: An empty Dict instance.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict[str, int].new()
        >>> data
        Dict()
        >>> # Equivalent to:
        >>> data: dict[str, int] = {}
        >>> data
        {}

        ```
        """
        return cls({})

    @staticmethod
    def from_kwargs[U](**kwargs: U) -> Dict[str, U]:
        """Create a `Dict` from keyword arguments.

        Args:
            **kwargs (U): Key-value pairs to initialize the Dict.

        Returns:
            Dict[str, U]: A new Dict instance containing the provided key-value pairs.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Dict.from_kwargs(a=1, b=2)
        Dict('a': 1, 'b': 2)

        ```
        """
        return Dict.from_ref(kwargs)

    @staticmethod
    def from_object(obj: object) -> Dict[str, Any]:
        """Create a `Dict` from an object `__dict__` attribute.

        We can't know in advance the values types, so we use `Any`.

        Args:
            obj (object): The object whose `__dict__` attribute will be used to create the Dict.

        Returns:
            Dict[str, Any]: A new Dict instance containing the attributes of the object.

        Example:
        ```python
        >>> import pyochain as pc
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Alice", 30)
        >>> pc.Dict.from_object(person)
        Dict('name': 'Alice', 'age': 30)

        ```
        """
        return Dict.from_ref(obj.__dict__)

    def insert(self, key: K, value: V) -> Option[V]:
        """Insert a key-value pair into the `Dict`.

        If the `Dict` did not have this **key** present, `NONE` is returned.

        If the `Dict` did have this **key** present, the **value** is updated, and the old value is returned.

        The **key** is not updated.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Option[V]: The previous value associated with the key, or None if the key was not present.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict.new()
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
        from ._option import Option

        previous = self._inner.get(key, None)
        self._inner[key] = value
        return Option(previous)

    def try_insert(self, key: K, value: V) -> Result[V, KeyError]:
        """Tries to insert a key-value pair into the map, and returns a `Result[V, KeyError]` containing the value in the entry (if successful).

        If the map already had this key present, nothing is updated, and an error containing the occupied entry and the value is returned.

        Args:
            key (K): The key to insert.
            value (V): The value associated with the key.

        Returns:
            Result[V, KeyError]: Ok containing the value if the key was not present, or Err containing a KeyError if the key already existed.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> d = pc.Dict.new()
        >>> d.try_insert(37, "a").unwrap()
        'a'
        >>> d.try_insert(37, "b")
        Err(KeyError('Key 37 already exists with value a.'))

        ```
        """
        from ._result import Err, Ok

        if key in self._inner:
            return Err(
                KeyError(f"Key {key} already exists with value {self._inner[key]}.")
            )
        self._inner[key] = value
        return Ok(value)

    def remove(self, key: K) -> Option[V]:
        """Remove a key from the `Dict` and return its value if it existed.

        Equivalent to `dict.pop(key, None)`, with an `Option` return type.

        Args:
            key (K): The key to remove.

        Returns:
            Option[V]: The value associated with the removed key, or None if the key was not present.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.remove(1)
        Some('a')
        >>> data.remove(3)
        NONE

        ```
        """
        from ._option import Option

        return Option(self._inner.pop(key, None))

    def remove_entry(self, key: K) -> Option[tuple[K, V]]:
        """Remove a key from the `Dict` and return the item if it existed.

        Return an `Option[tuple[K, V]]` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[tuple[K, V]]: The (key, value) pair associated with the removed key, or None if the key was not present.
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b"})
        >>> data.remove_entry(1)
        Some((1, 'a'))
        >>> data.remove_entry(3)
        NONE

        ```
        """
        from ._option import NONE, Some

        if key in self._inner:
            return Some((key, self._inner.pop(key)))
        return NONE

    def keys_iter(self) -> Iter[K]:
        """Return an `Iter` of the dict's keys.

        Returns:
            Iter[K]: An Iter wrapping the dictionary's keys.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).keys_iter().collect()
        Seq(1,)

        ```
        """
        from ._iter import Iter

        return Iter(self._inner.keys())

    def values_iter(self) -> Iter[V]:
        """Return an `Iter` of the `Dict` values.

        Returns:
            Iter[V]: An Iter wrapping the dictionary's values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).values_iter().collect()
        Seq(2,)

        ```
        """
        from ._iter import Iter

        return Iter(self._inner.values())

    @override
    def iter(self) -> Iter[tuple[K, V]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return an `Iter` of the dict's items.

        Yield tuples of (key, value) pairs.

        This is equivalent to calling `dict.items().__iter__()`, except the Iterator returned is wrapped in a pyochain `Iter`.

        `Iter.map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
        This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Note:
            Overrides `PyoIterable.iter` to correspond to `HashMap.iter()` Rust behavior.
            `Dict.__iter__` still yields keys only, as per Python convention.

        Returns:
            Iter[tuple[K, V]]: An `Iter` wrapping the dictionary's (key, value) pairs.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({"a": 1, "b": 2})
        >>> data.iter().collect()
        Seq(('a', 1), ('b', 2))
        >>> data.iter().map_star(lambda key, value: key).collect()
        Seq('a', 'b')
        >>> data.iter().map_star(lambda key, value: value).collect()
        Seq(1, 2)

        ```
        """
        from ._iter import Iter

        return Iter(self._inner.items())

    def get_item(self, key: K) -> Option[V]:
        """Retrieve a value from the `Dict`.

        Returns `Some(value)` if the key exists, or `None` if it does not.

        Args:
            key (K): The key to look up.

        Returns:
            Option[V]: Value that is associated with the key, or None if not found.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = {"a": 1}
        >>> pc.Dict(data).get_item("a")
        Some(1)
        >>> pc.Dict(data).get_item("x").unwrap_or('Not Found')
        'Not Found'

        ```
        """
        from ._option import Option

        return Option(self._inner.get(key, None))

    def is_empty(self) -> bool:
        """Returns true if the `Dict` contains no elements.

        Returns:
            bool: True if the Dict is empty, False otherwise.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> d = pc.Dict.new()
        >>> d.is_empty()
        True
        >>> d.insert(1, "a")
        NONE
        >>> d.is_empty()
        False

        ```
        """
        return len(self._inner) == 0
