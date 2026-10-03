from __future__ import annotations

import itertools
import multiprocessing
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedList as SortedListPy

from pyochain import Range, Seq
from pyochain.collections import SortedList

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

type IntoIter[T] = Callable[[T], Iterator[object]]
type IntoBoundedIter[T] = Callable[[T, int, int], Iterator[object]]
type List = list[int] | SortedList[int] | SortedListPy[int]
LIST_CLASSES = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(SortedListPy, id="sortedcontainers"),
        pytest.param(list),
        pytest.param(SortedList, id="pyochain"),
    ),
)


def _method_param[T](cls: type[T], f: Callable[[T], object]) -> ParameterSet:
    return pytest.param(cls, f, id=f"{cls.__module__}.{cls.__name__}.{f.__name__}")


BOUNDED_PARAMS = pytest.mark.parametrize(
    ("cls", "f"),
    ((
        _method_param(SortedListPy, SortedListPy[int].irange),
        _method_param(SortedListPy, SortedListPy[int].islice),
        _method_param(SortedList, SortedList[int].irange),
        _method_param(SortedList, SortedList[int].islice),
        pytest.param(list, itertools.islice),
    )),
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
    with pytest.raises(StopIteration):
        _ = next(it)


@BOUNDED_PARAMS
def test_bounded_clear[T: List](cls: type[T], f: IntoBoundedIter[T]) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: 10**9 + i))
    it = f(sl, 0, 10**9 + length)
    _ = next(it)
    sl.clear()
    with pytest.raises(StopIteration):
        _ = next(it)


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
    with pytest.raises(StopIteration):
        _ = next(it)


@LIST_CLASSES
@INTO_ITER_PARAMS
def test_clear_then_update[T: List](cls: type[T], into_iter: IntoIter[T]) -> None:
    sl = cls((1, 2, 3))
    it = into_iter(sl)
    _ = next(it)
    sl.clear()
    _update(sl, [4])
    with pytest.raises(StopIteration):
        _ = next(it)


@BOUNDED_PARAMS
def test_bounded_clear_then_update[T: List](
    cls: type[T], f: IntoBoundedIter[T]
) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: 10**9 + i))
    it = f(sl, 0, 10**9 + length)
    _ = next(it)
    sl.clear()
    _update(sl, [10**9 + length])
    with pytest.raises(StopIteration):
        _ = next(it)


def _update(sl: List, values: Iterable[int]) -> None:
    match sl:
        case list() | SortedList():
            sl.extend(values)
        case _:
            sl.update(values)
