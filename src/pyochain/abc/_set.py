from __future__ import annotations

from abc import ABC
from collections.abc import MutableSet
from collections.abc import Set as AbstractSet
from typing import override

from ._collection import PyoCollection


class PyoSet[T](PyoCollection[T], AbstractSet[T], ABC):
    """Extends `PyoCollection[T]` and `collections.abc.Set[T]`.

    Is the shared ABC for concrete set-like collections: `Set` and `FrozenSet`.

    Any concrete subclass must implement the required `Set` dunder methods:

    - `__contains__`
    - `__iter__`
    - `__len__`

    Example:
        ```python
        >>> from pyochain.abc import PyoSet
        >>> class MySet(PyoSet[int]):
        ...     def __init__(self, data: set[int]):
        ...         self._data = data
        ...
        ...     def __contains__(self, item: int) -> bool:
        ...         return item in self._data
        ...
        ...     def __iter__(self) -> Iterator[int]:
        ...         return iter(self._data)
        ...
        ...     def __len__(self) -> int:
        ...         return len(self._data)
        >>>
        >>> my_set = MySet({10, 20, 30})
        >>> my_set.is_subset({10, 20, 30, 40})
        True
        >>> my_set.is_superset({10})
        True
        >>> my_set.iter().sort()
        Vec(10, 20, 30)

        ```
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def all_unique(self) -> bool:
        return True

    def is_subset(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of this set are in `other` (including equality).

        Returns `True` if every element in this set is also present in `other`.

        This includes the case where both sets are identical.

        Use `is_subset_strict()` to exclude the equality case.

        Args:
            other (AbstractSet[T]): The set to check containment against.

        Returns:
            bool: `True` if all elements are contained, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> Set((1, 2)).is_subset({1, 2, 3})  # All elements present
            True
            >>> Set((1, 2)).is_subset({1, 2})  # Also True: they're equal
            True
            >>> Set((1, 4)).is_subset({1, 2, 3})  # 4 is not in the other set
            False

            ```
        """
        return self <= other

    def is_subset_strict(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of this set are in `other`, excluding equality.

        Returns `True` if every element in this set is also present in `other`, AND `other` contains at least one element not in this set.

        This is a proper (or strict) subset relation.

        Use `is_subset()` if you want to accept equal sets as well.

        Args:
            other (AbstractSet[T]): The set to check strict containment against.

        Returns:
            bool: `True` if this is a strict subset, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> Set((1, 2)).is_subset_strict({1, 2, 3})  # Proper subset
            True
            >>> Set((1, 2)).is_subset_strict({1, 2})  # Equal, not proper
            False
            >>> Set((1, 4)).is_subset_strict({1, 2, 3})  # 4 not contained
            False

            ```
        """
        return self < other

    def eq(self, other: AbstractSet[T]) -> bool:
        """Test whether this set contains exactly the same elements as `other`.

        Sets are equal if they have the same number of elements and every element in one is present in the other.

        Order is irrelevant for sets.

        This is an explicit method; you can also use the `==` operator directly.

        Args:
            other (AbstractSet[T]): The set to compare with.

        Returns:
            bool: `True` if both sets contain identical elements, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> Set((1, 2)).eq({2, 1})  # Same elements, different order
            True
            >>> Set((1, 2)).eq({1, 2, 3})  # Different number of elements
            False
            >>> Set((1, 2)).eq({1, 2})  # Identical
            True

            ```
        """
        return self == other

    def is_superset(self, other: AbstractSet[T]) -> bool:
        """Test whether all elements of `other` are in this set (including equality).

        Returns `True` if this set contains every element from `other`.

        This is the inverse of `is_subset()` -> if A is a subset of B, then B is a superset of A.

        Use `is_superset_strict()` (if available) to exclude equality.

        Args:
            other (AbstractSet[T]): The set to check containment for.

        Returns:
            bool: `True` if all elements from `other` are present, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> Set((1, 2, 3)).is_superset({1, 2})  # Contains all
            True
            >>> Set((1, 2)).is_superset({1, 2})  # Also True: they're equal
            True
            >>> Set((1, 2)).is_superset({1, 2, 3})  # Missing element 3
            False

            ```
        """
        return self >= other

    def is_disjoint(self, other: AbstractSet[T]) -> bool:
        """Test whether this set and `other` have no elements in common.

        Returns `True` if the intersection of the two sets is empty.

        This is the opposite of having any overlap.

        Args:
            other (AbstractSet[T]): The set to compare with.

        Returns:
            bool: `True` if no common elements exist, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Set
            >>> Set((1, 2)).is_disjoint((3, 4))  # No overlap
            True
            >>> Set((1, 2)).is_disjoint((2, 3))  # Share element 2
            False
            >>> Set((1, 2)).is_disjoint((1, 2))  # Identical sets
            False

            ```
        """
        return self.isdisjoint(other)

    def intersection(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing only elements present in both sets.

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
        return self._from_iterable(self & other)

    def union(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set containing all unique elements from both sets.

        This operation is commutative: `A.union(B) == B.union(A)`.

        Args:
            other (AbstractSet[T]): The set to combine with.

        Returns:
            AbstractSet[T]: A new set containing all elements from **self** and **other**.

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
        return self._from_iterable(self | other)

    def difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in this set but not in `other`.

        The result contains every element that is in this set EXCEPT those that are also present in `other`.

        This operation is NOT commutative.

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
        return self._from_iterable(self - other)

    def symmetric_difference(self, other: AbstractSet[T]) -> AbstractSet[T]:
        """Create a new set with elements in either set but not in both.

        The result contains elements that are in this set XOR `other`—i.e., elements present in one set but not in both.

        This is the opposite of [`Set::intersection`][Set.intersection].

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
        return self._from_iterable(self ^ other)


class PyoMutableSet[T](PyoSet[T], MutableSet[T], ABC):
    """ABCs for read-only and mutable sets."""

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]
