from __future__ import annotations

from typing import TYPE_CHECKING

from sortedcontainers import SortedList as PySortedList

from pyochain import Iter, Range
from pyochain.collections import SortedList

from ._utils import (
    BOUNDED_PARAMS,
    LOAD,
    REVERSE_PARAM,
    IterStop,
    List,
    SliceFn,
    assert_stop_iter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@REVERSE_PARAM
@BOUNDED_PARAMS
def test_pop[T: List](cls: type[T], f: SliceFn[T], *, reverse: bool) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: length + i))
    it = f(sl, 0, length * 2, reverse=reverse)
    _ = next(it)
    for _ in range(length - 1):
        _ = sl.pop()
    match sl:
        case PySortedList():
            assert_stop_iter(it, IndexError)
        case _:
            assert_stop_iter(it)


@BOUNDED_PARAMS
def test_remove_below_lower_bound[T: List](cls: type[T], f: SliceFn[T]) -> None:
    remove_range = 100
    stop = LOAD * 3
    sl = cls(range(stop))
    it = f(sl, LOAD - remove_range, stop, reverse=True)
    for value in range(remove_range):
        sl.remove(value)
    match sl:
        case list():
            assert_stop_iter(it)
        case PySortedList():
            _iter_after_removed_lower_bound(it, IndexError)
        case SortedList():
            _iter_after_removed_lower_bound(it, StopIteration)


def _iter_after_removed_lower_bound(it: Iterator[object], err: IterStop) -> None:
    iterator = Iter(it).skip(LOAD * 2 - 1)
    assert iterator.next().is_some()
    assert_stop_iter(iterator, err)
