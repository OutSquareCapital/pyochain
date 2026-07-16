from collections.abc import Iterable, Iterator, MutableMapping
from typing import Self, overload, override

from _typeshed import SupportsGetItem, SupportsKeysAndGetItem

from ._types import DictConvertible
from .abc import PyoMutableMapping, PyoReversible

type IntoDict[K, V] = dict[K, V] | Dict[K, V]

class Dict[K, V](PyoMutableMapping[K, V], PyoReversible[K]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    Accept the same input types as the built-in `dict`, including `Mapping`, `Iterable` of key-value pairs, and objects implementing `__getitem__()` and `keys()`.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.

    Args:
        data (DictConvertible[K, V]): Initial data for the Dict that can converted to a dictionary.

    See Also:
        - [`Dict::from_ref`][from_ref]: Create a `Dict` from an existing dictionary, no-copy.
        - [`Dict::from_kwargs`][from_kwargs]: Create a `Dict` from keyword arguments.
        - [`Dict::from_object`][from_object]: Create a `Dict` from an object's `__dict__` attribute, no-copy.

    Example:
        The most straightforward way to create a `Dict` is from a standard Python `dict`.

        This will copy the data, just like the built-in `dict` constructor.
        ```python
        >>> from pyochain import Dict
        >>> py_dict = {1: "a", 2: "b"}
        >>> pyochain_dict = Dict(py_dict)
        >>> pyochain_dict
        Dict(1: 'a', 2: 'b')

        ```
        Another common case is when you have an iterable of key-value pairs, such as the one returned by `dict::items`, or an `Iterator` of tuples.
        ```python
        >>> from pyochain import Dict, Iter, Seq
        >>>
        >>> names = ("alice", "bob", "charlie", "dave")
        >>> ages = (30, 25, 35, 40)
        >>> records = Iter(names).zip(ages).collect(Dict)
        >>> records
        Dict('alice': 30, 'bob': 25, 'charlie': 35, 'dave': 40)
        >>> records.items().iter().collect(Seq)
        Seq(('alice', 30), ('bob', 25), ('charlie', 35), ('dave', 40))

        ```
        Any object that implements the `Mapping` protocol can also be directly converted to a `Dict`:
        ```python
        >>> from collections.abc import Mapping, Iterator, Iterable
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
        ...
        ...     def keys(self) -> Iterable[int]:
        ...         return iter(self._data)
        ...
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

    def __init__(self, data: DictConvertible[K, V]) -> None: ...
    @classmethod
    def from_keys[K1, V1](cls, keys: Iterable[K1], value: V1 = None) -> Dict[K1, V1]:
        """Create a `Dict` from an iterable of keys, all mapped to the same value.

        This is the equivalent of `dict.fromkeys`, but returns a `Dict` instance.

        Args:
            keys (Iterable[K1]): An iterable of keys to include in the mapping.
            value (V1): The value that each key will be mapped to.

        Returns:
            Dict[K1, V1]: A new `Dict` instance containing the specified keys and value.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> Dict.from_keys([1, 2, 3], "a")
            Dict(1: 'a', 2: 'a', 3: 'a')
            >>> Dict.from_keys("abc")
            Dict('a': None, 'b': None, 'c': None)

            ```
        """

    @override
    def __iter__(self) -> Iterator[K]: ...
    @override
    def __contains__(self, key: object) -> bool: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __getitem__(self, key: K) -> V: ...
    @override
    def __setitem__(self, key: K, value: V) -> None: ...
    @override
    def __delitem__(self, key: K) -> None: ...
    @override
    def __eq__(self, other: object) -> bool: ...
    def __or__[T1, T2](self, value: IntoDict[T1, T2], /) -> Dict[K | T1, V | T2]: ...
    def __ror__[T1, T2](self, value: IntoDict[T1, T2], /) -> Dict[K | T1, V | T2]: ...
    def __ior__(
        self, value: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]], /
    ) -> Self: ...
    @override
    def __reversed__(self) -> Iterator[K]: ...
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

    @staticmethod
    def from_object(obj: object) -> Dict[str, object]:
        """Create a `Dict` from an object `__dict__` attribute.

        We can't know in advance the values types, so we use `object`.

        Syntactic sugar for `Dict.from_ref(obj.__dict__)`.

        Warning:
            This take a direct reference to the object's `__dict__`, so any modifications to the resulting `Dict` will also affect the original object's attributes, and vice versa.

        Args:
            obj (object): The object whose `__dict__` attribute will be used to create the `Dict`.

        Returns:
            Dict[str, object]: A new `Dict` instance containing the attributes of the object.

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
    def copy(self) -> Dict[K, V]:
        """Create a shallow copy of the `Dict`.

        Returns:
            Dict[K, V]: The copied `Dict` instance.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d1 = Dict({1: "a", 2: "b"})
            >>> d2 = d1.copy()
            >>> d2
            Dict(1: 'a', 2: 'b')
            >>> d1 is d2
            False
            >>> d1.inner is d2.inner
            False

            ```
        """

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    @override
    def pop[T](self, key: K, default: T | None = None, /) -> V | T | None: ...
    def union[T1, T2](self, other: IntoDict[T1, T2]) -> Dict[K | T1, V | T2]:
        """Merge another `dict` or `Dict` with this `Dict`, returning a new one with the combined key-value pairs.

        If there are duplicate keys, the values from *other* will overwrite those in `Self`.

        This is equivalent to `|` on a standard Python `dict`.

        Args:
            other (IntoDict[T1, T2]): The other mapping to merge with.

        Returns:
            Dict[K | T1, V | T2]: A new mapping containing the merged key-value pairs.

        See Also:
            - [`Dict::union_mut`][union_mut]: Merge another mapping into `Self` in-place.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d1 = Dict({1: "a", 2: "b"})
            >>> d2 = Dict({2: "c", 3: "d"})
            >>> d3 = d1.union(d2)
            >>> d3
            Dict(1: 'a', 2: 'c', 3: 'd')
            >>> d1 is d3 or d2 is d3
            False

            ```
        """

    def union_mut(
        self, other: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]]
    ) -> Self:
        """Merge another `dict` or `Dict` into `Self` in-place.

        If there are duplicate keys, the values from *other* will overwrite those in `Self`.

        This is equivalent to `|=` on a standard Python `dict`.

        Args:
            other (SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]]): The other mapping to merge with.

        Returns:
            Self: The modified `Dict` instance after merging.

        See Also:
            - [`Dict::union`][union]: Merge another mapping with `Self` in a new `Dict`.
            - `Dict::update` to accept any compatible `Iterable`.

        Example:
            ```python
            >>> from pyochain import Dict
            >>> d1 = Dict({1: "a", 2: "b"})
            >>> d1_inner_id = id(d1.inner)
            >>> d2 = Dict({2: "c", 3: "d"})
            >>> d1.union_mut(d2)
            Dict(1: 'a', 2: 'c', 3: 'd')
            >>> id(d1.inner) == d1_inner_id
            True
            >>> d1.union_mut(((4, "e"), (5, "f")))
            Dict(1: 'a', 2: 'c', 3: 'd', 4: 'e', 5: 'f')
            >>> id(d1.inner) == d1_inner_id
            True

            ```
        """

    @override
    def popitem(self) -> tuple[K, V]: ...
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
    def update(self, m: object = None, /, **kwargs: V) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def setdefault[T](
        self: MutableMapping[K, T | None], key: K, default: None = None, /
    ) -> T | None: ...
    @overload
    def setdefault(self, key: K, default: V, /) -> V: ...
    @overload
    def setdefault[T](self, key: K, default: object = None, /) -> V: ...
    @override
    def setdefault[T](self, key: K, default: object = None, /) -> V | T | None: ...
