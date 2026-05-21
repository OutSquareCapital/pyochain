from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MutableSet,
    ValuesView,
)
from collections.abc import Set as AbstractSet
from typing import Self, override

from ._utils import get_repr
from .abc import PyoCollection, PyoMappingView, PyoSet


class BaseConcreteSet[T](ABC):
    """Internal mixin for concrete set-like classes.

    It provides a common interface for documentation, DRY code, and type consistency, as well as enforcing a *you must implement concrete methods for operator dunders* policy for set operations.

    pyochain philosophy is to prefer explicit method calls that read like natural language instead of obscure operators.

    Concrete set classes should inherit from this base class and implement the following abstract methods:

        - `intersection()`
        - `union()`
        - `difference()`
        - `symmetric_difference()`
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @abstractmethod
    def intersection(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing only elements present in both sets.

        The result contains every element that exists in both this set and `other`.

        If the sets have no common elements, the result is empty.

        This operation is commutative: `A.intersection(B) == B.intersection(A)`.

        Args:
            other (AbstractSet[T]): The set to intersect with.

        Returns:
            AbstractSet[T]: A new `Set` containing shared elements only.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> from_set = Set((1, 2))
            >>> from_set.intersection({2, 3})
            Set(2,)
            >>> from_set.intersection({3, 4})
            Set()
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().intersection({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('b', 'c')
            >>> from_items = (
            ...     dct.items()
            ...     .intersection({("b", 2), ("c", 3), ("d", 4)})
            ...     .iter()
            ...     .sort()
            ... )
            >>> from_items
            Vec(('b', 2), ('c', 3))

            ```
        """

    @abstractmethod
    def union(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing all unique elements from both sets.

        The result includes every element from this set and every element from `other`.

        Duplicates are automatically removed.

        This operation is commutative: `A.union(B) == B.union(A)`.

        Args:
            other (AbstractSet[T]): The set to combine with.

        Returns:
            AbstractSet[T]: A new set containing all elements from both sets.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).union({2, 3}).union({4}).iter().sort()
            Vec(1, 2, 3, 4)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().union({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('a', 'b', 'c', 'd')
            >>> from_items = (
            ...     dct.items().union({("b", 2), ("c", 3), ("d", 4)}).iter().sort()
            ... )
            >>> from_items
            Vec(('a', 1), ('b', 2), ('c', 3), ('d', 4))

            ```
        """

    @abstractmethod
    def difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in this set but not in `other`.

        The result contains every element that is in this set EXCEPT those that are also present in `other`.

        This operation is NOT commutative.

        Use `r_difference()` if you need the reversed argument order for pipelines.

        Args:
            other (AbstractSet[T]): The set whose elements should be excluded.

        Returns:
            AbstractSet[T]: A new set containing elements unique to this set.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).difference({2, 3})
            Set(1,)
            >>> Set((1, 2)).difference({3, 4}).iter().sort()
            Vec(1, 2)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = dct.keys().difference({"b", "c", "d"}).iter().sort()
            >>> from_keys
            Vec('a')
            >>> from_items = (
            ...     dct.items().difference({("b", 2), ("c", 3), ("d", 4)}).iter().sort()
            ... )
            >>> from_items
            Vec(('a', 1))

            ```
        """

    @abstractmethod
    def symmetric_difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in either set but not in both.

        The result contains elements that are in this set XOR `other`—i.e., elements present in one set but not in both.

        This is the opposite of intersection.

        This operation is commutative: `A.symmetric_difference(B) == B.symmetric_difference(A)`.

        Args:
            other (AbstractSet[T]): The set to compute symmetric difference with.

        Returns:
            AbstractSet[T]: A new set containing elements unique to each set.

        Example:
            ```python
            >>> from pyochain import Set, Dict
            >>> Set((1, 2)).symmetric_difference({2, 3}).iter().sort()
            Vec(1, 3)
            >>> Set((1, 2, 3)).symmetric_difference({3, 4, 5}).iter().sort()
            Vec(1, 2, 4, 5)
            >>> dct = Dict.from_ref({"a": 1, "b": 2, "c": 3})
            >>> from_keys = (
            ...     dct.keys().symmetric_difference({"b", "c", "d"}).iter().sort()
            ... )
            >>> from_keys
            Vec('a', 'd')
            >>> from_items = (
            ...     dct.items()
            ...     .symmetric_difference({("b", 2), ("c", 3), ("d", 4)})
            ...     .iter()
            ...     .sort()
            ... )
            >>> from_items
            Vec(('a', 1), ('d', 4))

            ```
        """


class Set[T](PyoSet[T], BaseConcreteSet[T]):
    """`Set` represent an in- memory **unordered**  collection of **unique** elements.

    Implements the `Collection` Protocol from `collections.abc`, so it can be used as a standard immutable collection.

    The underlying data structure is a `frozenset`.

    Tip:
        - `Set(frozenset)` is a no-copy operation since Python optimizes this under the hood.
        - If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
            data (Iterable[T]): The data to initialize the Set with.

    Example:
    ```python
    >>> from pyochain import Set
    >>> Set(())
    Set()
    >>> s = Set((1, 2, 2, 3))
    >>> s
    Set(1, 2, 3)
    >>> s_2 = Set(s.inner)
    >>> # No copy is made when creating s_2 from s.inner, they reference the same underlying frozenset.
    >>> is_no_copy = (
    ...     s.inner is s_2.inner
    ...     and s.inner is s.inner
    ...     and s_2.inner is s.inner
    ...     and frozenset(s.inner) is s.inner
    ... )
    >>> is_no_copy
    True
    >>> # However, creating a new Set from s (not using .inner) will be a copy operation.
    >>> Set(s).inner is s.inner
    False

    ```
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    __match_args__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute]
    _inner: frozenset[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = frozenset(data)

    @property
    def inner(self) -> frozenset[T]:
        """Get the underlying `frozenset` data structure.

        Useful when interoperating with functions that require a standard Python `frozenset`.

        Returns:
            frozenset[T]: The underlying frozenset.
        """
        return self._inner

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner)})"

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._inner

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def intersection(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner & other)

    @override
    def union(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner | other)

    @override
    def difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner - other)

    @override
    def symmetric_difference(self, other: AbstractSet[T]) -> Self:
        return self.__class__(self._inner ^ other)


class PyoValuesView[V](ValuesView[V], PyoMappingView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the values in a pyochain mapping.

    This concrete class is returned by the `PyoMapping.values()` method, and inherits from `collections.abc.ValuesView` and `PyoCollection`.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoKeysView[K](KeysView[K], PyoMappingView[K], PyoSet[K], BaseConcreteSet[K]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the keys in a pyochain mapping.

    This concrete class is returned by the `PyoMapping.keys()` method, and inherits from `collections.abc.KeysView`, `PyoMappingView`, `PyoSet`, and `BaseConcreteSet`.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self & other)

    @override
    def union(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self ^ other)


class PyoItemsView[K, V](  # pyright: ignore[reportUnsafeMultipleInheritance]
    ItemsView[K, V],
    PyoMappingView[tuple[K, V]],
    PyoSet[tuple[K, V]],
    BaseConcreteSet[tuple[K, V]],
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    This concrete class is returned by the `PyoMapping.items()` method, and inherits from `collections.abc.ItemsView`, `PyoMappingView`, `PyoSet`, and `BaseConcreteSet`.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping.items()`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self & other)

    @override
    def union(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference(
        self, other: AbstractSet[tuple[K, V]]
    ) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self ^ other)


class SetMut[T](BaseConcreteSet[T], MutableSet[T], PyoCollection[T]):
    """A mutable `set` wrapper with functional API.

    Unlike `Set` which is immutable, `SetMut` allows in-place modification of elements.

    Implement the `MutableSet` interface, so elements can be modified in place, and passed to any function/object expecting a standard mutable `set`.

    Underlying data structure is a `set`.

    Tip:
        If you have an existing `set`, prefer using `SetMut.from_ref()` to avoid unnecessary copying.

    Args:
        data (Iterable[T]): The mutable set to wrap.
    """

    __slots__ = ("_inner",)  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]
    _inner: set[T]

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = set(data)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def __contains__(self, item: object) -> bool:
        return item in self._inner

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({get_repr(self._inner)})"

    @property
    def inner(self) -> set[T]:
        """Get the underlying `set` data structure.

        Useful when interoperating with functions that require a standard Python `set`.

        Returns:
            set[T]: The underlying set.
        """
        return self._inner

    @staticmethod
    def from_ref[V](data: set[V]) -> SetMut[V]:
        """Create a `SetMut` from a reference to an existing `set`.

        This method wraps the provided `set` without copying it, allowing for efficient object instanciation.

        This is the recommended way to create a `SetMut` from foreign functions that return `set` objects.

        Warning:
            Since the `SetMut` directly references the original `set`, any modifications made to the `SetMut` will also affect the original `set`, and vice versa.

        Args:
            data (set[V]): The `set` to wrap.

        Returns:
            SetMut[V]: A new `SetMut` instance wrapping the provided `set`.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> original_set = {1, 2, 3}
            >>> set_obj = SetMut.from_ref(original_set)
            >>> set_obj
            SetMut(1, 2, 3)
            >>> original_set.add(4)
            >>> set_obj
            SetMut(1, 2, 3, 4)

            ```
        """
        instance: SetMut[V] = object.__new__(SetMut)  # pyright: ignore[reportUnknownVariableType]
        instance._inner = data
        return instance

    @override
    def add(self, value: T) -> None:
        """Add an element to **self**.

        Args:
            value (T): The element to add.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s = SetMut(("a", "b"))
            >>> s.add("c")
            >>> s.iter().sort()
            Vec('a', 'b', 'c')

            ```
        """
        self._inner.add(value)

    @override
    def discard(self, value: T) -> None:
        """Remove an element from **self** if it is a member.

        Unlike `.remove()`, the `discard()` method does not raise an exception when an element is missing from the set.

        Args:
            value (T): The element to remove.

        Example:
            ```python
            >>> from pyochain import SetMut
            >>> s = SetMut(("a", "b", "c"))
            >>> s.discard("b")
            >>> s.iter().sort()
            Vec('a', 'c')

            ```
        """
        self._inner.discard(value)

    @override
    def intersection(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner & other)

    @override
    def union(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner | other)

    @override
    def difference(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner - other)

    @override
    def symmetric_difference(self, other: AbstractSet[T]) -> SetMut[T]:
        return self.from_ref(self._inner ^ other)
