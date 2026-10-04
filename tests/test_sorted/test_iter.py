from __future__ import annotations

import multiprocessing

import pytest
from sortedcontainers import SortedList as SortedListPy

from pyochain import Iter, Range, Seq
from pyochain.collections import SortedList

from ._utils import (
    IntoIter,
    List,
    assert_stop_iter,
    stop_iter_or_unsupported,
    update_list,
)

LIST_CLASSES = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(SortedListPy, id="sortedcontainers"),
        pytest.param(list),
        pytest.param(SortedList, id="pyochain"),
    ),
)


INTO_ITER_PARAMS = pytest.mark.parametrize("into_iter", (iter, reversed))


def test_iterator_pointer() -> None:
    """Can crash if we don't correctly handle references with unsafe impls.

    A crash kills the pytest process silently, hence the child process.
    """
    proc = multiprocessing.Process(target=_consume_orphan_iter)
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def _consume_orphan_iter() -> None:
    _ = Range(10).pipe(SortedList).iter().collect(list)


@LIST_CLASSES
@INTO_ITER_PARAMS
def test_clear[T: List](cls: type[T], into_iter: IntoIter[T]) -> None:
    sl = cls((1, 2, 3))
    it = into_iter(sl)
    sl.clear()
    assert_stop_iter(it)


@LIST_CLASSES
@INTO_ITER_PARAMS
def test_pop[T: List](cls: type[T], into_iter: IntoIter[T]) -> None:
    data = Seq(1, 2, 3, 4, 5)
    sl = cls(data)
    it = into_iter(sl)
    elem = next(it)
    assert elem == data[0] or elem == data[-1]
    for _ in range(data.len() - 1):
        _ = sl.pop()
    assert_stop_iter(it)


@LIST_CLASSES
@INTO_ITER_PARAMS
def test_clear_then_update[T: List](cls: type[T], into_iter: IntoIter[T]) -> None:
    sl = cls((1, 2, 3))
    it = into_iter(sl)
    _ = next(it)
    sl.clear()
    update_list(sl, [4])
    stop_iter_or_unsupported(sl, it)


@LIST_CLASSES
def test_add_after_last_consumed[T: List](cls: type[T]) -> None:
    stop = 10
    sl = cls(range(stop))
    it = iter(sl)
    _ = Iter(it).take(stop).last()
    update_list(sl, [stop])
    assert next(it) == stop
    assert_stop_iter(it)


@LIST_CLASSES
def test_remove_consumed_in_chunk[T: List](cls: type[T]) -> None:
    start = 1000
    stop = 3000
    sl = cls(range(stop))
    it = iter(sl)
    _ = Iter(it).take(start - 1).last()
    sl.remove(0)
    assert tuple(it) == tuple(range(start, stop))


@LIST_CLASSES
def test_remove_far_chunk_rev[T: List](cls: type[T]) -> None:
    start = 3000
    stop = 1000
    sl = cls(range(start))
    it = reversed(sl)
    _ = Iter(it).take(stop).last()
    for value in range(stop):
        sl.remove(value)
    assert tuple(it) == tuple(range(start - 1, stop - 1, -1))
