from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

# TODO: Theses types are manually extracted from typeshed and rewritten in modern python style
# This is error prone, because we can very easily miss overloads, as well as being tedious¨
# We should handle this automatically with a dedicated script, if possible.


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


type SupportsAnyAdd = SupportsAdd[Any, Any]


class SupportsRAdd[T, T1](Protocol):
    def __radd__(self, x: T, /) -> T1: ...


class SupportsKeysAndGetItem[K, V](Protocol):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


class SupportsSumWithNoDefaultGiven(
    SupportsAdd[Any, Any], SupportsRAdd[int, Any], Protocol
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

type PositiveInteger = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
]
type NegativeInteger = Literal[
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    -7,
    -8,
    -9,
    -10,
    -11,
    -12,
    -13,
    -14,
    -15,
    -16,
    -17,
    -18,
    -19,
    -20,
]
type LiteralInteger = PositiveInteger | NegativeInteger | Literal[0]
