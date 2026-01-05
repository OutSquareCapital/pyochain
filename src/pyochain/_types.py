from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from .traits import Pipeable

if TYPE_CHECKING:
    from ._iter import Iter

# transformed iterables result types


@dataclass(slots=True)
class Unzipped[T, V](Pipeable):
    """Represents the result of unzipping an iterator of pairs into two separate iterators.

    See `Iter.unzip()` for details.
    """

    left: Iter[T]
    """An iterator over the first elements of the pairs."""
    right: Iter[V]
    """An iterator over the second elements of the pairs."""


@dataclass(slots=True)
class Peekable[T](Pipeable):
    """Represents the result of peeking into an iterator.

    See `Iter.peekable()` for details.
    """

    peek: Iter[T]
    """An iterator over the peeked elements."""
    values: Iter[T]
    """An iterator of values, still including the peeked elements."""


# Iterations result types


class Item[K, V](NamedTuple):
    """Represents a key-value pair from a `Dict`."""

    key: K
    """The key of the item."""
    value: V
    """The value associated with the key."""

    def __repr__(self) -> str:
        return f"({self.key.__repr__()}, {self.value.__repr__()})"


class Enumerated[T](NamedTuple):
    """Represents an item with its associated index in an enumeration.

    See `Iter.enumerate()` for details.
    """

    idx: int
    """The index of the item in the enumeration."""
    value: T
    """The value of the item."""

    def __repr__(self) -> str:
        return f"({self.idx}, {self.value.__repr__()})"


class Group[K, V](NamedTuple):
    """Represents a grouping of values by a common key.

    See `Iter.group_by()` for details.
    """

    key: K
    """The common key for the group."""
    values: Iter[V]
    """An `Iter` over the values associated with the key."""

    def __repr__(self) -> str:
        return f"({self.key.__repr__()}, {self.values.__repr__()})"


# typeshed protocols


class SupportsDunderLT[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...


class SupportsDunderGT[T](Protocol):
    def __gt__(self, other: T, /) -> bool: ...


class SupportsDunderLE[T](Protocol):
    def __le__(self, other: T, /) -> bool: ...


class SupportsDunderGE[T](Protocol):
    def __ge__(self, other: T, /) -> bool: ...


class SupportsAdd[T, T1](Protocol):
    def __add__(self, x: T, /) -> T1: ...


class SupportsRAdd[T, T1](Protocol):
    def __radd__(self, x: T, /) -> T1: ...


class SupportsKeysAndGetItem[K, V](Protocol):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


class SupportsSumWithNoDefaultGiven[T](
    SupportsAdd[T, Any], SupportsRAdd[int, T], Protocol
): ...


type SupportsRichComparison[T] = SupportsDunderLT[T] | SupportsDunderGT[T]
