"""Benchmarks for pyochain vs sortedcontainers comparison.

Reference:

https://grantjenks.com/docs/sortedcontainers/performance.html
"""

from __future__ import annotations

import random
from math import sqrt
from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList

from pyochain import Range, Seq
from pyochain.collections import SortedKeyList, SortedList

from ._utils import SIZES, shuffled

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from ._utils import BenchFixture

type List = SortedList[int] | PySortedList[int]
type KeyList = type[SortedKeyList[int, int] | PySortedKeyList[int, int]]
type SetupOut[L] = tuple[tuple[L], dict[str, object]]
CLS_PARAMS = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(PySortedList, id="sortedcontainers"),
        pytest.param(SortedList, id="pyochain"),
    ),
)
SIZE_PARAMS = pytest.mark.parametrize("size", SIZES)


# NOTE: mostly test the "empty maxes" path in `update` method.
@CLS_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: type[List], size: int) -> None:
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
def test_contains(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__contains__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__getitem__).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem_slice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl[i : i + 1]).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_left(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_left).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_right(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_right).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(obj: Iterable[int]) -> None:
        for _ in obj:
            pass

    sl = cls(range(size))
    benchmark(f, sl)


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_collect(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    sl = cls(range(size))
    _ = benchmark(tuple, sl)


@CLS_PARAMS
@SIZE_PARAMS
def test_iter_rev_collect(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(lst: Sequence[int]) -> Sequence[int]:
        return tuple(reversed(lst))

    _ = benchmark(f, cls(Range(size)))


@CLS_PARAMS
@SIZE_PARAMS
def test_count(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.count(i) == 1).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_index(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.index(i) == i).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_islice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    sl = cls(range(size))
    assert benchmark(lambda: tuple(sl.islice(start=0, stop=size - 1, reverse=False)))


@CLS_PARAMS
@SIZE_PARAMS
def test_irange(benchmark: BenchFixture, cls: type[List], size: int) -> None:
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
def test_update_large(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = shuffled(size)
    if cls is PySortedList:
        benchmark(lambda: cls([1]).update(data))
    else:
        benchmark(lambda: cls([1]).extend(data))


@CLS_PARAMS
@SIZE_PARAMS
def test_update_small[L: List](
    benchmark: BenchFixture, cls: type[L], size: int
) -> None:
    k = size // 8
    values = shuffled(k)
    if cls is PySortedList:
        run_pedantic(benchmark, cls, size, lambda sl: sl.update(values))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownLambdaType]
    else:
        run_pedantic(benchmark, cls, size, lambda sl: sl.extend(values))


@CLS_PARAMS
@SIZE_PARAMS
def test_add(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)

    def f(sl: List) -> None:
        for val in values:
            sl.add(val)

    run_pedantic(benchmark, cls, size, f)


@CLS_PARAMS
@SIZE_PARAMS
def test_remove(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)

    def f(sl: List) -> None:
        for val in values:
            sl.remove(val)

    run_pedantic(benchmark, cls, size, f)


@CLS_PARAMS
@SIZE_PARAMS
def test_pop(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    calls = range(_calls_nb(size))

    def f(sl: List) -> None:
        for _ in calls:
            _ = sl.pop()

    run_pedantic(benchmark, cls, size, f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    positions = _del_positions(size)

    def f(sl: List) -> None:
        for pos in positions:
            del sl[pos]

    run_pedantic(benchmark, cls, size, f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delslice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    positions = _del_positions(size)

    def f(sl: List) -> None:
        for pos in positions:
            del sl[pos : pos + 1]

    run_pedantic(benchmark, cls, size, f)


def _del_positions(size: int) -> Seq[int]:
    return (
        Range(_calls_nb(size))
        .iter()
        .map(lambda i: random.randrange(size - i))
        .collect(Seq)
    )


def _values_sample(size: int) -> Seq[int]:
    return shuffled(size).iter().take(_calls_nb(size)).collect(Seq)


def run_pedantic[L: List](
    benchmark: BenchFixture, cls: type[L], size: int, f: Callable[[L], object]
) -> None:

    reps = Range(_calls_nb(SIZES.last()) // _calls_nb(size))
    rounds = max(20, int(50 * sqrt(SIZES.last() // max(size, 4_000))))

    def run(lists: Seq[L]) -> None:
        for sl in lists:
            _ = f(sl)

    def setup() -> SetupOut[Seq[L]]:
        return (reps.iter().map(lambda _: cls(range(size))).collect(Seq),), {}

    benchmark.pedantic(run, setup=setup, rounds=rounds, warmup_rounds=rounds // 20)


def _calls_nb(size: int) -> int:
    return min(size // 4, 1_000)
