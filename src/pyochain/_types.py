from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
