from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import pytest
from sortedcontainers import SortedList as SortedListPy

from pyochain.collections import SortedList

type IntoIter[T] = Callable[[T], Iterator[object]]
type List = list[int] | SortedList[int] | SortedListPy[int]


def update_list(sl: List, values: Iterable[int]) -> None:
    match sl:
        case list() | SortedList():
            sl.extend(values)
        case _:
            sl.update(values)


def stop_iter_or_unsupported(sl: List, it: Iterator[object]) -> None:
    """In some cases, `sortedcontainers` have undefined behavior, whilst `pyochain` emulate Python `list`."""
    match sl:
        case SortedListPy():
            _ = next(it)
        case _:
            assert_stop_iter(it)


def assert_stop_iter(it: Iterator[object]) -> None:
    with pytest.raises(StopIteration):
        _ = next(it)
