from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from pyochain.abc import Checkable, PyoIterable

@runtime_checkable
class PyoContainer[T](Checkable, Protocol):
    """ABC for `collections.abc.Container` Protocol."""
    @abstractmethod
    def __contains__(self, x: T, /) -> bool: ...
    def contains(self, value: T) -> bool:
        """Check if the `Container` contains the specified **value**.

        This is equivalent to `value in self`, but as a method.

        Args:
            value (T): The value to check for existence.

        Returns:
            bool: True if the value exists in the Collection, False otherwise.

        Example:
            ```python
            from pyochain import Dict

            data = Dict(a=1, b=2)

            assert data.contains("a")
            assert not data.contains("c")
            ```
        """

@runtime_checkable
class PyoSized(Checkable, Protocol):
    @abstractmethod
    def __len__(self) -> int: ...
    def len(self) -> int:
        """Return the length of `Self`.

        Equivalent to `len(self)`, but as a method.

        Returns:
            int: The number of elements in `Self`.

        Example:
            ```python
            from pyochain import Dict

            assert Dict(a=1, b=2).len() == 2
            ```
        """

    def is_empty(self) -> bool:
        """Returns `True` if the `Collection` contains no elements.

        Returns:
            bool: `True` if the `Collection` is empty, `False` otherwise.

        Example:
            ```python
            from pyochain import Dict

            d = Dict(())
            assert d.is_empty()
            d.insert(1, "a")
            assert not d.is_empty()
            ```
        """

@runtime_checkable
class PyoCollection[T](PyoIterable[T], PyoContainer[Any], PyoSized, Protocol):
    """`Extends `PyoIterable[T]` and `collections.abc.Collection[T]`.

    This includes `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, etc...

    Any concrete subclass must implement the required `Collection` dunder methods:

    - `__iter__`
    - `__len__`
    - `__contains__`
    """
