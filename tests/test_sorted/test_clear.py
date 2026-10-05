from __future__ import annotations

from typing import TYPE_CHECKING

from sortedcontainers import SortedList as PySortedList

from pyochain import Iter, Range, Some
from pyochain.collections import SortedList

from ._utils import (
    BOUNDED_PARAMS,
    INTO_ITER_PARAMS,
    LIST_CLASSES,
    REVERSE_PARAM,
    UPDATE_PARAMS,
    IntoIter,
    List,
    assert_stop_iter,
    update_list,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._utils import SliceFn


@UPDATE_PARAMS
@LIST_CLASSES
@INTO_ITER_PARAMS
def test_iter[T: List](cls: type[T], into_iter: IntoIter[T], *, update: bool) -> None:
    sl = cls((1, 2, 3))
    it = into_iter(sl)
    sl.clear()
    if update:
        update_list(sl, [4])
    match sl, into_iter.__name__ == "iter", update:
        case (_, _, False) | (list() | SortedList(), False, True):
            assert_stop_iter(it)
        case (PySortedList(), _, True) | (list() | SortedList(), True, True):
            assert next(it) == 4
            assert_stop_iter(it)


@UPDATE_PARAMS
@REVERSE_PARAM
@BOUNDED_PARAMS
def test_slice[T: List](
    cls: type[T], f: SliceFn[T], *, reverse: bool, update: bool
) -> None:
    length = 10
    stop = length * 2
    sl = cls(Range(length).iter().map(lambda i: length + i))
    it = f(sl, 0, stop, reverse=reverse)
    sl.clear()
    if update:
        update_list(sl, [stop])
    match sl, reverse, update:
        case PySortedList(), True, _:
            assert _last_val_is_some(it, length)
            assert_stop_iter(it)
        case PySortedList(), False, _:
            assert _last_val_is_some(it, stop - 1)
            assert_stop_iter(it)
        case list() | SortedList(), False, True:
            assert next(it) == stop
            assert_stop_iter(it)
        case list() | SortedList(), _, _:
            assert_stop_iter(it)


def _last_val_is_some(it: Iterator[object], length: int) -> bool:
    return Iter(it).skip(length - 1).next() == Some(length)
