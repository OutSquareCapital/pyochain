from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .traits import PyoMutableMapping

if TYPE_CHECKING:
    from ._types import DictConvertible


class Dict[K, V](PyoMutableMapping[K, V]):
    """A `Dict` is a key-value store similar to Python's built-in `dict`, but with additional methods inspired by Rust's `HashMap`.

    Accept the same input types as the built-in `dict`, including `Mapping`, `Iterable` of key-value pairs, and objects implementing `__getitem__()` and `keys()`.

    Implement the `MutableMapping` interface, so all standard dictionary operations are supported.

    Tip:
        Prefer using `Dict.from_ref` when wrapping existing dictionaries to avoid unnecessary copying.

    Args:
        data (DictConvertible[K, V]): Initial data for the Dict that can converted to a dictionary.
    """

    __slots__ = ("_inner",)
    _inner: dict[K, V]

    def __init__(self, data: DictConvertible[K, V]) -> None:
        self._inner = dict(data)

    def __repr__(self) -> str:
        from pprint import pformat

        return (
            f"{self.__class__.__name__}({pformat(self._inner, sort_dicts=False)[1:-1]})"
        )

    def __iter__(self) -> Iterator[K]:
        return iter(self._inner)

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

        Warning:
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
        >>> dict_obj.insert(1, "z")
        Some('a')
        >>> original_dict
        {1: 'z', 2: 'b', 3: 'c'}

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
