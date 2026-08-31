from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedList as PySortedList

from pyochain import Range
from pyochain.collections import SortedList as PyoSortedList

from ._utils import SIZES

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._utils import BenchFixture

type SortedList = type[PyoSortedList[int] | PySortedList[int]]
CLS_PARAMS = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(PySortedList, id="sortedcontainers"),
        pytest.param(PyoSortedList, id="pyochain"),
    ),
)
SIZE_PARAMS = pytest.mark.parametrize("size", SIZES)


@CLS_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    assert benchmark(cls, r)  # pyright: ignore[reportArgumentType]


@CLS_PARAMS
@SIZE_PARAMS
def test_contains(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(sl.__contains__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(sl.__getitem__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_left(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(sl.bisect_left).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_right(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(sl.bisect_right).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    def f(obj: Iterable[int]) -> None:
        for _ in obj:
            pass

    r = Range(size)
    sl = cls(r)
    benchmark(f, sl)


@CLS_PARAMS
@SIZE_PARAMS
def test_count(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(lambda i: sl.count(i) == 1).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_index(benchmark: BenchFixture, cls: SortedList, size: int) -> None:
    r = Range(size)
    sl = cls(r)
    assert benchmark(lambda: r.iter().map(lambda i: sl.index(i) == i).last())
