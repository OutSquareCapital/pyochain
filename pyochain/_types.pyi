from collections.abc import Callable, Hashable, Iterable
from typing import Any, Literal, Protocol

from _typeshed import SupportsRichComparison

# TODO: Theses types are manually extracted from typeshed and rewritten in modern python style
# This is error prone, because we can very easily miss overloads, as well as being tedious¨
# We should handle this automatically with a dedicated script, if possible.

type SupportsHashableAndRichComparison = (
    SupportsHashableAndDunderLT[Any] | SupportsHashableAndDunderGT[Any]
)

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]

class SupportsDunderLT[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...

class SupportsHashableAndDunderLT[T](Hashable, SupportsDunderLT[T], Protocol): ...

class SupportsDunderGT[T](Protocol):
    def __gt__(self, other: T, /) -> bool: ...

class SupportsHashableAndDunderGT[T](Hashable, SupportsDunderGT[T], Protocol): ...

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
