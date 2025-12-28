from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, NamedTuple, Self, TypeIs, overload

import cytoolz as cz

from ._config import get_config
from ._core import Pipeable

if TYPE_CHECKING:
    from ._core import SupportsKeysAndGetItem
    from ._iter import Iter
    from ._option import Option
    from ._result import Result


class Item[K, V](NamedTuple):
    """Represents a key-value pair from a `Dict`."""

    key: K
    """The key of the item."""
    value: V
    """The value associated with the key."""

    def __repr__(self) -> str:
        return f"({self.key.__repr__()}, {self.value.__repr__()})"


class Dict[K, V](Pipeable, MutableMapping[K, V]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    You can initialize it with an existing Python `dict`, or from any object that can be converted into a dict with the `from_` method.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.

    """

    __slots__ = ("_inner",)

    _inner: dict[K, V]

    def __init__(
        self, data: Mapping[K, V] | Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]
    ) -> None:
        self._inner = dict(data)

    def __repr__(self) -> str:
        return f"{self.into(lambda d: get_config().dict_repr(d._inner))}"

    def __iter__(self) -> Iterator[K]:
        return self._inner.__iter__()

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, key: K) -> V:
        return self._inner[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._inner[key] = value

    def __delitem__(self, key: K) -> None:
        del self._inner[key]

    def contains_key(self, key: K) -> bool:
        """Check if the `Dict` contains the specified key.

        This is equivalent to using the `in` keyword directly on the `Dict`.

        Args:
            key (K): The key to check for existence.

        Returns:
            bool: True if the key exists in the Dict, False otherwise.
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

    def length(self) -> int:
        """Return the number of key-value pairs in the `Dict`.

        Equivalent to `len(self)`.

        Returns:
            int: The number of items in the Dict.

        ```python
        >>> import pyochain as pc
        >>> data = pc.Dict({1: "a", 2: "b", 3: "c"})
        >>> data.length()
        3

        ```
        """
        return len(self._inner)

    @classmethod
    def new(cls) -> Self:
        """Create an empty `Dict`.

        Returns:
            Self: An empty Dict instance.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict.new()
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
        {'a': 1, 'b': 2}
        """
        return Dict(kwargs)

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
        return Option.from_(previous)

    def try_insert(self, key: K, value: V) -> Result[V, KeyError]:
        """Tries to insert a key-value pair into the map, and returns a mutable reference to the value in the entry.

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

        return Option.from_(self._inner.pop(key, None))

    def remove_entry(self, key: K) -> Option[Item[K, V]]:
        """Remove a key from the `Dict` and return the `Item` if it existed.

        Return an `Item` containing the (key, value) pair if the key was present.

        Args:
            key (K): The key to remove.

        Returns:
            Option[Item[K, V]]: The (key, value) pair associated with the removed key, or None if the key was not present.
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
            return Some(Item(key, self._inner.pop(key)))
        return NONE

    def keys_iter(self) -> Iter[K]:
        """Return an Iter of the dict's keys.

        Returns:
            Iter[K]: An Iter wrapping the dictionary's keys.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).keys_iter().collect()
        Seq(1,)

        ```
        """
        from ._iter import Iter

        return self.into(lambda d: Iter(d._inner.keys()))

    def values_iter(self) -> Iter[V]:
        """Return an Iter of the dict's values.

        Returns:
            Iter[V]: An Iter wrapping the dictionary's values.

        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).values_iter().collect()
        Seq(2,)

        ```
        """
        from ._iter import Iter

        return self.into(lambda d: Iter(d._inner.values()))

    def iter(self) -> Iter[Item[K, V]]:
        """Return an `Iter` of the dict's items.

        Yield `Item` instances representing each (key, value) pair in the `Dict`.

        Returns:
            Iter[Item[K, V]]: An Iter wrapping the dictionary's (key, value) pairs.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({"a": 1, "b": 2}).iter().collect()
        Seq(('a', 1), ('b', 2))

        ```
        """
        from ._iter import Iter

        return self.into(lambda d: Iter(Item(*it) for it in d._inner.items()))

    def get_item(self, key: K) -> Option[V]:
        """Retrieve a value from the `Dict`.

        Returns `Some(value)` if the key exists, or `None` if it does not.

        Args:
            key (K): The key to look up.

        Returns:
            Option[V]: Value that is associated with the key, or None if not found.

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

        return Option.from_(self._inner.get(key, None))

    def is_empty(self) -> bool:
        """Returns true if the map contains no elements.

        Returns:
            bool: True if the Dict is empty, False otherwise.

        Examples:
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

    @overload
    def filter_keys[U](self, predicate: Callable[[K], TypeIs[U]]) -> Dict[U, V]: ...
    @overload
    def filter_keys(self, predicate: Callable[[K], bool]) -> Dict[K, V]: ...
    def filter_keys[U](
        self,
        predicate: Callable[[K], bool | TypeIs[U]],
    ) -> Dict[K, V] | Dict[U, V]:
        """Return a new `Dict` containing only the items whose keys satisfy the **predicate**.

        This does not modify the original `Dict`.

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
        return Dict(cz.dicttoolz.keyfilter(predicate, self._inner))

    @overload
    def filter_values[U](self, predicate: Callable[[V], TypeIs[U]]) -> Dict[K, U]: ...
    @overload
    def filter_values(self, predicate: Callable[[V], bool]) -> Dict[K, V]: ...
    def filter_values[U](
        self,
        predicate: Callable[[V], bool] | Callable[[V], TypeIs[U]],
    ) -> Dict[K, V] | Dict[K, U]:
        """Return a new `Dict` containing only the items whose values satisfy the **predicate**.

        This does not modify the original `Dict`.

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
        return Dict(cz.dicttoolz.valfilter(predicate, self._inner))

    def filter_items(self, predicate: Callable[[tuple[K, V]], bool]) -> Dict[K, V]:
        """Return a new `Dict` containing only the items that satisfy the **predicate**.

        This does not modify the original `Dict`.

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
        return Dict(cz.dicttoolz.itemfilter(predicate, self._inner))

    def map_keys[T](self, func: Callable[[K], T]) -> Dict[T, V]:
        """Return a new `Dict` with keys transformed by the provided **func**.

        The values of the original `Dict` remain unchanged.

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
        return Dict(cz.dicttoolz.keymap(func, self._inner))

    def map_values[T](self, func: Callable[[V], T]) -> Dict[K, T]:
        """Return a new `Dict` with values transformed by the provided **func**.

        The keys of the original `Dict` remain unchanged.

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
        return Dict(cz.dicttoolz.valmap(func, self._inner))

    def map_items[KR, VR](
        self,
        func: Callable[[tuple[K, V]], tuple[KR, VR]],
    ) -> Dict[KR, VR]:
        """Return a new `Dict` with items transformed by the provided **func**.

        The original `Dict` remains unchanged.

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
        return Dict(cz.dicttoolz.itemmap(func, self._inner))
