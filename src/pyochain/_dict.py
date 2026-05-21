from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self, override

from .abc import PyoMutableMapping

if TYPE_CHECKING:
    from ._types import DictConvertible


class Dict[K, V](PyoMutableMapping[K, V]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    Accept the same input types as the built-in `dict`, including `Mapping`, `Iterable` of key-value pairs, and objects implementing `__getitem__()` and `keys()`.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.

    Tip:
        Prefer using `Dict::from_ref` when wrapping existing dictionaries to avoid unnecessary copying.

    Args:
        data (DictConvertible[K, V]): Initial data for the Dict that can converted to a dictionary.

    See Also:
        - `Dict::from_ref`: Create a `Dict` from an existing dictionary, no-copy.
        - `Dict::from_kwargs`: Create a `Dict` from keyword arguments.
        - `Dict::from_object`: Create a `Dict` from an object's `__dict__` attribute, no-copy.

    Example:
        The most straightforward way to create a `Dict` is from a standard Python `dict`. This will copy the data, just like the built-in `dict` constructor.
        ```python
        >>> from pyochain import Dict
        >>> py_dict = {1: "a", 2: "b"}
        >>> pyochain_dict = Dict(py_dict)
        >>> pyochain_dict
        Dict(1: 'a', 2: 'b')

        ```
        Another common case is when you have an iterable of key-value pairs, such as the one returned by `dict::items`, or an `Iterator` of tuples.
        ```python
        >>> from pyochain import Dict, Iter
        >>> names = ("alice", "bob", "charlie", "dave")
        >>> ages = (30, 25, 35, 40)
        >>> records = Iter(names).zip(ages).collect(Dict)
        >>> records
        Dict('alice': 30, 'bob': 25, 'charlie': 35, 'dave': 40)
        >>> records.items().iter().collect()
        Seq(('alice', 30), ('bob', 25), ('charlie', 35), ('dave', 40))

        ```
        Any object that implements the `Mapping` protocol can also be directly converted to a `Dict`:
        ```python
        >>> from collections.abc import Mapping
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class CustomMapping(Mapping[int, str]):
        ...     data: dict[int, str]
        ...
        ...     def __getitem__(self, key: int) -> str:
        ...         return self.data[key]
        ...
        ...     def __iter__(self) -> Iterator[int]:
        ...         return iter(self.data)
        ...
        ...     def __len__(self) -> int:
        ...         return len(self.data)
        >>> custom_mapping = CustomMapping({1: "a", 2: "b"})
        >>> Dict(custom_mapping)
        Dict(1: 'a', 2: 'b')

        ```
        But it can also be as minimal as an object that implements `__getitem__` and `keys`:
        ```python
        >>> from pyochain import Dict
        >>>
        >>> class MinimalDictLike:
        ...     def __init__(self, data: dict[int, str]) -> None:
        ...         self._data = data
        ...     def keys(self) -> Iterable[int]:
        ...         return iter(self._data)
        ...     def __getitem__(self, key: int) -> str:
        ...         return self._data[key]
        >>>
        >>> minimal_dict_like = MinimalDictLike({1: "a", 2: "b"})
        >>> Dict(minimal_dict_like)
        Dict(1: 'a', 2: 'b')

        ```
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: dict[K, V]

    def __init__(self, data: DictConvertible[K, V]) -> None:
        self._inner = dict(data)

    @override
    def __repr__(self) -> str:
        from pprint import pformat

        return (
            f"{self.__class__.__name__}({pformat(self._inner, sort_dicts=False)[1:-1]})"
        )

    @property
    def inner(self) -> dict[K, V]:
        """Get the underlying `dict` data structure.

        Useful when interoperating with functions that require a standard Python `dict`.

        Returns:
            dict[K, V]: The underlying dictionary.
        """
        return self._inner

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def __getitem__(self, key: K) -> V:
        return self._inner[key]

    @override
    def __setitem__(self, key: K, value: V) -> None:
        self._inner[key] = value

    @override
    def __delitem__(self, key: K) -> None:
        del self._inner[key]

    @staticmethod
    def from_ref[K1, V1](data: dict[K1, V1]) -> Dict[K1, V1]:
        """Wrap an existing Python builtin `dict` without copying.

        This is the recommended way to create a `Dict` from foreign functions that return a standard Python `dict`.

        Warning:
            Any modifications made to this `Dict` will also affect the original data structure, and vice versa.

        Args:
            data (dict[K1, V1]): The dictionary to wrap.

        Returns:
            Dict[K1, V1]: A new `Dict` instance wrapping the provided dictionary.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> original_dict = {1: "a", 2: "b", 3: "c"}
            >>> ref_dict = Dict.from_ref(original_dict)
            >>> ref_dict
            Dict(1: 'a', 2: 'b', 3: 'c')
            >>> ref_dict.insert(1, "z")
            Some('a')
            >>> original_dict
            {1: 'z', 2: 'b', 3: 'c'}
            >>> ref_dict.inner is original_dict
            True

            ```
        """
        instance: Dict[K1, V1] = Dict.__new__(Dict)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @staticmethod
    def from_kwargs[U](**kwargs: U) -> Dict[str, U]:
        """Create a `Dict` from keyword arguments.

        Args:
            **kwargs (U): Key-value pairs to initialize the Dict.

        Returns:
            Dict[str, U]: A new Dict instance containing the provided key-value pairs.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> Dict.from_kwargs(a=1, b=2)
            Dict('a': 1, 'b': 2)

            ```
        """
        return Dict.from_ref(kwargs)

    @staticmethod
    def from_object(obj: object) -> Dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Create a `Dict` from an object `__dict__` attribute.

        We can't know in advance the values types, so we use `Any`.

        Warning:
            This take a direct reference to the object's `__dict__`, so any modifications to the resulting `Dict` will also affect the original object's attributes, and vice versa.

        Args:
            obj (object): The object whose `__dict__` attribute will be used to create the Dict.

        Returns:
            Dict[str, Any]: A new Dict instance containing the attributes of the object.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            >>>
            >>> person = Person("Alice", 30)
            >>> pyo_dict = Dict.from_object(person)
            >>> pyo_dict
            Dict('name': 'Alice', 'age': 30)
            >>> pyo_dict.inner is person.__dict__
            True
            >>> pyo_dict.insert("name", "Bob")
            Some('Alice')
            >>> person
            Person(name='Bob', age=30)

            ```
        """
        return Dict.from_ref(obj.__dict__)

    def merge(self, other: dict[K, V] | Self) -> Dict[K, V]:
        """Merge another `dict` or `Dict` with this `Dict`, returning a new one with the combined key-value pairs.

        If there are duplicate keys, the values from *other* will overwrite those in `Self`.

        This is a **copy** operation. If you want to merge in-place, use the `update` method instead.

        Args:
            other (dict[K, V] | Self): The other mapping to merge with.

        Returns:
            Dict[K, V]: A new mapping containing the merged key-value pairs.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d1 = Dict({1: "a", 2: "b"})
            >>> d2 = Dict({2: "c", 3: "d"})
            >>> d3 = d1.merge(d2)
            >>> d3
            Dict(1: 'a', 2: 'c', 3: 'd')
            >>> d1 is d3 or d2 is d3
            False

            ```
        """
        match other:
            case Dict():
                new = self._inner | other._inner
            case dict():
                new = self._inner | other
        return self.from_ref(new)
