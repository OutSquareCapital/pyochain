from collections.abc import Iterable, Sized
from typing import Protocol


class SupportsDunderLT[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...


class SupportsDunderGT[T](Protocol):
    def __gt__(self, other: T, /) -> bool: ...


class SupportsDunderLE[T](Protocol):
    def __le__(self, other: T, /) -> bool: ...


class SupportsDunderGE[T](Protocol):
    def __ge__(self, other: T, /) -> bool: ...


class SupportsKeysAndGetItem[K, V](Protocol):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


class SupportsAllComparisons[T](
    SupportsDunderLT[T],
    SupportsDunderGT[T],
    SupportsDunderLE[T],
    SupportsDunderGE[T],
    Protocol,
): ...


type SupportsRichComparison[T] = SupportsDunderLT[T] | SupportsDunderGT[T]


class SizedIterable[T](Sized, Iterable[T]): ...
