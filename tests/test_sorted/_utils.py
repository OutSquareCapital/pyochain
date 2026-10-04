from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedDict as SortedDictPy
from sortedcontainers import SortedList as SortedListPy
from sortedcontainers import SortedSet as SortedSetPy

from pyochain.collections import SortedDict, SortedList, SortedSet

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

type IntoIter[T] = Callable[[T], Iterator[object]]
type List = list[int] | SortedList[int] | SortedListPy[int]
type AnySet = set[int] | SortedSet[int] | SortedSetPy[int]
type Dict = dict[int, object] | SortedDict[int, object] | SortedDictPy[int, object]


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


def method_param[T](cls: type[T], f: Callable[[T], object]) -> ParameterSet:
    return pytest.param(cls, f, id=f"{cls.__module__}.{cls.__name__}.{f.__name__}")
