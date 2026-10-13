from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList

from pyochain import Range
from pyochain.collections import SortedKeyList, SortedList

from ._utils import SIZES, shuffled

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ._utils import BenchFixture

type List = type[SortedList[int] | PySortedList[int]]
type KeyList = type[SortedKeyList[int, int] | PySortedKeyList[int, int]]
CLS_PARAMS = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(PySortedList, id="sortedcontainers"),
        pytest.param(SortedList, id="pyochain"),
    ),
)
SIZE_PARAMS = pytest.mark.parametrize("size", SIZES)


@CLS_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: List, size: int) -> None:
    def f(x: Iterable[int]) -> Sequence[int]:
        return cls(x)

    assert benchmark(f, shuffled(size))


@CLS_PARAMS
@SIZE_PARAMS
def test_init_by_key(benchmark: BenchFixture, cls: KeyList, size: int) -> None:
    data = shuffled(size)
    if cls is PySortedKeyList:
        assert benchmark(_py_key_list, data)
    else:
        assert benchmark(_key_list, data)


def _py_key_list(x: Iterable[int]) -> PySortedKeyList[int, int]:
    return PySortedKeyList(x, key=lambda x: x + 1)


def _key_list(x: Iterable[int]) -> SortedKeyList[int, int]:
    return SortedKeyList(lambda x: x + 1, x)


@CLS_PARAMS
@SIZE_PARAMS
def test_contains(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__contains__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__getitem__).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem_slice(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl[i : i + 1]).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_left(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_left).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_right(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_right).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: List, size: int) -> None:
    def f(obj: Iterable[int]) -> None:
        for _ in obj:
            pass

    sl = cls(range(size))
    benchmark(f, sl)


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_collect(benchmark: BenchFixture, cls: List, size: int) -> None:
    sl = cls(range(size))
    _ = benchmark(tuple, sl)


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_rev_collect(benchmark: BenchFixture, cls: List, size: int) -> None:
    def f(lst: Sequence[int]) -> Sequence[int]:
        return tuple(reversed(lst))

    _ = benchmark(f, cls(Range(size)))


@CLS_PARAMS
@SIZE_PARAMS
def test_count(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.count(i) == 1).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_index(benchmark: BenchFixture, cls: List, size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.index(i) == i).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_islice(benchmark: BenchFixture, cls: List, size: int) -> None:
    sl = cls(range(size))
    assert benchmark(lambda: tuple(sl.islice(start=0, stop=size - 1, reverse=False)))


@CLS_PARAMS
@SIZE_PARAMS
def test_irange(benchmark: BenchFixture, cls: List, size: int) -> None:
    sl = cls(range(size))
    assert benchmark(
        lambda: tuple(
            sl.irange(
                minimum=0, maximum=size - 1, inclusive=(True, True), reverse=False
            )
        )
    )


@CLS_PARAMS
@SIZE_PARAMS
def test_add_pop(benchmark: BenchFixture, cls: List, size: int) -> None:
    r = Range(size)
    sl = cls()
    assert benchmark(lambda: r.iter().map(sl.add).map(lambda _: sl.pop(-1)).last())
