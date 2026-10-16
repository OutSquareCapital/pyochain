"""Methods dispatchs to handle API divergences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList

from pyochain.collections import SortedKeyList, SortedList

from ._utils import KeyFn, KeyList, List

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def update[L: List](cls: type[L]) -> Callable[[L, Iterable[int]], None]:
    if cls is PySortedList:
        return PySortedList[int].update  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].extend  # pyright: ignore[reportReturnType]


def add[L: List](cls: type[L]) -> Callable[[L, int], None]:
    if cls is PySortedList:
        return PySortedList[int].add  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].append  # pyright: ignore[reportReturnType]


def key_init[L: KeyList](cls: type[L]) -> Callable[[Iterable[int], KeyFn], L]:
    if cls is PySortedKeyList:
        return lambda it, fn: PySortedKeyList[int, int](it, key=fn)  # pyright: ignore[reportReturnType]
    else:
        return lambda it, fn: SortedKeyList[int, int](fn, it)  # pyright: ignore[reportReturnType]
