from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

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


type SupportsComparison[T] = (
    SupportsDunderLE[T]
    | SupportsDunderGE[T]
    | SupportsDunderGT[T]
    | SupportsDunderLT[T]
)
type SupportsRichComparison[T] = SupportsDunderLT[T] | SupportsDunderGT[T]
type DictConvertible[K, V] = (
    Mapping[K, V] | Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]
)
