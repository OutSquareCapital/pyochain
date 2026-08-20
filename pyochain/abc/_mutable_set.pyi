from abc import abstractmethod
from collections.abc import MutableSet
from collections.abc import Set as AbstractSet
from typing import Any, Self, override

from pyochain.abc import PyoSet

class PyoMutableSet[T](PyoSet[T], MutableSet[T]):  # pyright: ignore[reportImplicitAbstractClass]
    """ABCs for read-only and mutable sets."""
    # TODO: check logic of typing and impl for vanilla collections.abc
    # Pyo3 forces incompatible implementations of those dunders -> they return `None` instead of `Self`.
    # However, typing-wise, they must return `Self`, otherwise type checkers will complain about unknown return types.
    @override
    def __ior__(self, it: AbstractSet[T], /) -> Self: ...
    @override
    def __iand__(self, it: AbstractSet[Any], /) -> Self: ...
    @override
    def __ixor__(self, it: AbstractSet[T], /) -> Self: ...
    @override
    def __isub__(self, it: AbstractSet[Any], /) -> Self: ...
    @abstractmethod
    @override
    def add(self, value: T, /) -> None: ...
    @abstractmethod
    @override
    def discard(self, value: T, /) -> None: ...

    # Mixin methods
    @override
    def clear(self) -> None:
        """Remove all elements from this set.

        Example:
            ```python
            from pyochain import SetMut

            s = SetMut(1, 2, 3)
            s.clear()
            assert s.len() == 0
            ```
        """

    @override
    def pop(self) -> T: ...
    @override
    def remove(self, value: T, /) -> None: ...
