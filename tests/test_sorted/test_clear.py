from __future__ import annotations

from typing import TYPE_CHECKING

from sortedcontainers import SortedList as PySortedList

from pyochain import Iter, Range
from pyochain.collections import SortedList

from ._utils import (
    BOUNDED_PARAMS,
    INTO_ITER_PARAMS,
    LIST_CLASSES,
    LOAD,
    REVERSE_PARAM,
    UPDATE_PARAMS,
    IntoIter,
    List,
    assert_stop_iter,
    update_list,
)

if TYPE_CHECKING:
    from ._utils import SliceFn


@UPDATE_PARAMS
@LIST_CLASSES
@INTO_ITER_PARAMS
def test_iter[T: List](cls: type[T], into_iter: IntoIter[T], *, update: bool) -> None:
    sl = cls((1, 2, 3))
    it = into_iter(sl)
    sl.clear()
    if update:
        update_list(sl, [LOAD])
    match sl, into_iter.__name__ == "iter", update:
        case (PySortedList() | SortedList() | list(), _, False) | (list(), False, True):
            assert_stop_iter(it)
        case (PySortedList() | SortedList(), _, True) | (list(), True, True):
            assert next(it) == LOAD
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
    it = Iter(f(sl, 0, stop, reverse=reverse))
    sl.clear()
    if update:
        update_list(sl, [LOAD])
    match sl, reverse, update:
        case PySortedList(), False, _:
            assert it.next().unwrap() == length
            assert it.skip(length - 2).next().unwrap() == stop - 1
            assert_stop_iter(it)
        case PySortedList(), True, _:
            assert it.next().unwrap() == stop - 1
            assert it.skip(length - 2).next().unwrap() == length
            assert_stop_iter(it)
        case list() | SortedList(), False, True:
            assert it.next().unwrap() == LOAD
            assert_stop_iter(it)
        case list() | SortedList(), _, _:
            assert_stop_iter(it)
