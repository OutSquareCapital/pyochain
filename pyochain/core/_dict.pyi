from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Self, overload, override

from _typeshed import SupportsGetItem, SupportsKeysAndGetItem

from pyochain.abc import PyoMutableMapping, PyoReversible
from pyochain.core.protocols import KwargsWrapper

type DictConvertible[K, V] = (
    Mapping[K, V] | Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]
)
type IntoDict[K, V] = dict[K, V] | Dict[K, V]

class Dict[K, V](PyoMutableMapping[K, V], PyoReversible[K], KwargsWrapper[K, V]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.
    """
    @overload
    def __new__(cls, iterable: DictConvertible[K, V], /) -> Dict[K, V]: ...
    @overload
    def __new__(cls, **kwargs: V) -> Dict[str, V]: ...
    @overload
    def __new__[K1, V1](
        cls, iterable: DictConvertible[str, V], **kwargs: V
    ) -> Dict[str, V]: ...
    def __new__(cls, iterable: DictConvertible[K, V] = (), **kwargs: V) -> Self:
        """Create a new `Dict` instance.

        Accept the same input types as the built-in `dict`, including `Mapping`, `Iterable` of key-value pairs, and objects implementing `__getitem__()` and `keys()`.

        Args:
            iterable (DictConvertible[K, V]): Initial data for the Dict that can converted to a dictionary.
            **kwargs (V): Additional key-value pairs to include in the Dict.

        Returns:
            Self: A new `Dict` instance containing the provided key-value pairs.

        See Also:
            - [`Dict::wrap`][wrap]: Create a `Dict` from an existing dictionary, no-copy.
            - [`Dict::of`][of]: Create a `Dict` from keyword arguments.
            - [`Dict::from_object`][from_object]: Create a `Dict` from an object's `__dict__` attribute, no-copy.

        Example:
            The most straightforward way to create a `Dict` is from a standard Python `dict`.

            This will copy the data, just like the built-in `dict` constructor.
            ```python
            from pyochain import Dict

            py_dict = {1: "a", 2: "b"}
            pyochain_dict = Dict(py_dict)
            assert pyochain_dict == Dict({1: "a", 2: "b"})
            ```
            Another common case is when you have an iterable of key-value pairs, such as the one returned by `dict::items`, or an `Iterator` of tuples.
            ```python
            from pyochain import Dict, Iter, Seq

            names = Seq("alice", "bob", "charlie", "dave")
            ages = (30, 25, 35, 40)
            records = names.iter().zip(ages).collect(Dict)
            assert records == Dict({"alice": 30, "bob": 25, "charlie": 35, "dave": 40})
            assert records.items().iter().collect(Seq) == (
                ("alice", 30),
                ("bob", 25),
                ("charlie", 35),
                ("dave", 40),
            )
            ```
            Any object that implements the `Mapping` protocol can also be directly converted to a `Dict`:
            ```python
            from collections.abc import Mapping, Iterator, Iterable
            from dataclasses import dataclass

            @dataclass
            class CustomMapping(Mapping[int, str]):
                data: dict[int, str]

                def __getitem__(self, key: int) -> str:
                    return self.data[key]

                def __iter__(self) -> Iterator[int]:
                    return iter(self.data)

                def __len__(self) -> int:
                    return len(self.data)

            custom_mapping = CustomMapping({1: "a", 2: "b"})
            assert Dict(custom_mapping) == Dict({1: "a", 2: "b"})
            ```
            But it can also be as minimal as an object that implements `__getitem__` and `keys`:
            ```python
            from pyochain import Dict

            class MinimalDictLike:
                def __init__(self, data: dict[int, str]) -> None:
                    self._data = data

                def keys(self) -> Iterable[int]:
                    return iter(self._data)

                def __getitem__(self, key: int) -> str:
                    return self._data[key]

            minimal_dict_like = MinimalDictLike({1: "a", 2: "b"})
            assert Dict(minimal_dict_like) == Dict({1: "a", 2: "b"})
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
            from pyochain import Dict

            d = Dict.from_keys(["a", "b", "c"], 1)
            assert d == Dict(a=1, b=1, c=1)
            d2 = Dict.from_keys("abc")
            assert d2 == Dict(a=None, b=None, c=None)
            ```
        """

    @override
    @staticmethod
    def from_iter(iterable: Iterable[tuple[K, V]], /) -> Dict[K, V]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    @staticmethod
    def wrap[K1, V1](data: dict[K1, V1]) -> Dict[K1, V1]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    @staticmethod
    def of[U](**kwargs: U) -> Dict[str, U]: ...
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
            from pyochain import Dict, Some
            from dataclasses import dataclass

            @dataclass
            class Person:
                name: str
                age: int

            person = Person("Alice", 30)
            pyo_dict = Dict.from_object(person)
            assert pyo_dict == Dict(name="Alice", age=30)
            assert pyo_dict.insert("name", "Bob") == Some("Alice")
            assert person == Person(name="Bob", age=30)
            ```
        """

    def copy(self) -> Dict[K, V]:
        """Create a shallow copy of the `Dict`.

        Returns:
            Dict[K, V]: The copied `Dict` instance.

        Example:
            ```python
            from pyochain import Dict

            d1 = Dict(a=1, b=2)
            d2 = d1.copy()
            assert d2 == Dict(a=1, b=2)
            assert d1 is not d2
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
            from pyochain import Dict

            d1 = Dict(a=1, b=2)
            d2 = Dict(c=2, d=3)
            d3 = d1.union(d2)
            assert d3 == Dict(a=1, b=2, c=2, d=3)
            assert d1 is not d3 and d2 is not d3
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
            from pyochain import Dict, Some

            d1 = Dict(a=1, b=2)
            d2 = Dict(c=2, d=3)
            d1.union_mut(d2)
            assert d1 == Dict(a=1, b=2, c=2, d=3)
            d1.union_mut((("e", 4), ("f", 5)))
            assert d1 == Dict(a=1, b=2, c=2, d=3, e=4, f=5)
            assert d1.insert("b", 100) == Some(2)
            assert d1 == Dict(a=1, b=100, c=2, d=3, e=4, f=5)
            assert d2 == Dict(c=2, d=3)
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
