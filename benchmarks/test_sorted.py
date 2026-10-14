"""Benchmarks for pyochain vs sortedcontainers comparison.

Reference:

https://grantjenks.com/docs/sortedcontainers/performance.html
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import sqrt
from random import Random
from typing import TYPE_CHECKING, Final

import pytest
from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList

from pyochain import Iter, Range, Seq, Vec
from pyochain.collections import SortedKeyList, SortedList

from ._utils import SIZES

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
RANDOM: Final[Random] = Random(0)


# NOTE: mostly test the "empty maxes" path in `update` method.
@CLS_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def f(x: Iterable[int]) -> Sequence[int]:
        return cls(x)

    assert benchmark(f, _shuffled(size))


@CLS_PARAMS
@SIZE_PARAMS
def test_init_by_key(benchmark: BenchFixture, cls: KeyList, size: int) -> None:
    data = _shuffled(size)
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
    data = _shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__contains__).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = _shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.__getitem__).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_getitem_slice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = _shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl[i : i + 1]).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_left(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = _shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(sl.bisect_left).last()) == data.last()


@CLS_PARAMS
@SIZE_PARAMS
def test_bisect_right(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = _shuffled(size)
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
    data = _shuffled(size)
    sl = cls(range(size))
    assert benchmark(lambda: data.iter().map(lambda i: sl.count(i) == 1).last())


@CLS_PARAMS
@SIZE_PARAMS
def test_index(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    data = _shuffled(size)
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
    data = _shuffled(size)
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
    values = _shuffled(k)
    config = BenchConfig(benchmark, cls, size)
    if cls is PySortedList:
        config.run_pedantic(lambda sl: sl.update(values))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownLambdaType]
    else:
        config.run_pedantic(lambda sl: sl.extend(values))


@CLS_PARAMS
@SIZE_PARAMS
def test_add(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)

    def f(sl: List) -> None:
        for val in values:
            sl.add(val)

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_remove(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    values = _values_sample(size)

    def f(sl: List) -> None:
        for val in values:
            sl.remove(val)

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_pop(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    calls = range(_calls_nb(size))

    def f(sl: List) -> None:
        for _ in calls:
            _ = sl.pop()

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    positions = _del_positions(size, _calls_nb(size), 1)

    def f(sl: List) -> None:
        for pos in positions:
            del sl[pos]

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_delslice(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    width = max(1, size // 10)
    pos = _del_positions(size, 1, width).first()

    def f(sl: List) -> None:
        del sl[pos : pos + width]

    BenchConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SIZE_PARAMS
def test_priority_queue(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def op(sl: List, value: int, pos: int) -> None:
        if pos < 40:
            sl.add(value)
        elif pos < 80:
            _ = sl.pop()
        elif pos < 90:
            sl.discard(value)
        elif pos < 99:
            _ = value in sl
        else:
            _ = Iter(sl).take(102).last()

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_multiset(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def op(sl: List, value: int, pos: int) -> None:
        if pos < 75:
            assert value in sl
        elif pos < 85:
            sl.add(value)
        elif pos < 95:
            sl.remove(value)
        elif value < len(sl):
            _ = sl[value]

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_ranking(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def op(sl: List, value: int, pos: int) -> None:
        if pos < 40:
            if value < len(sl):
                _ = sl[value]
        elif pos < 80:
            assert sl.index(value) >= 0
        elif pos < 90:
            sl.add(value)
        else:
            sl.remove(value)

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_neighbor(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    def op(sl: List, value: int, pos: int) -> None:
        if pos < 75:
            _ = sl.bisect_right(value)
        elif pos < 85:
            sl.add(value)
        elif pos < 95:
            sl.remove(value)
        else:
            _ = Iter(sl).take(102).last()

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_intervals(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    config = BenchConfig(benchmark, cls, size)

    def op(sl: List, value: int, pos: int) -> None:
        if pos < 30:
            _ = sl.bisect_right(value)
        elif pos < 50:
            if value < len(sl):
                _ = sl[value]
        elif pos < 70:
            if value < len(sl):
                del sl[value]
        elif pos < 80:
            if value < len(sl):
                other = RANDOM.randrange(value, min(value + 100, len(sl)))
                _ = sl[value:other]
        elif pos < 90:
            sl.add(value)
        else:
            sl.discard(value)

    config.run_workload(op)


@dataclass(slots=True)
class BenchConfig[L: List]:
    bench: BenchFixture
    cls: type[L]
    size: int

    def run_pedantic(self, f: Callable[[L], object]) -> None:
        reps = Range(_calls_nb(SIZES.last()) // _calls_nb(self.size))
        rounds = max(20, int(50 * sqrt(SIZES.last() // max(self.size, 4_000))))

        def run(lists: Seq[L]) -> None:
            for sl in lists:
                _ = f(sl)

        def setup() -> SetupOut[Seq[L]]:
            data = reps.iter().map(lambda _: self.cls(range(self.size))).collect(Seq)
            return (data,), {}

        self.bench.pedantic(run, setup=setup, rounds=rounds, warmup_rounds=rounds // 20)

    def run_workload(self, op: Callable[[List, int, int], object]) -> None:
        values = _shuffled(self.size).iter().step_by(10).collect(Seq)

        def f(sl: List) -> None:
            for value in values:
                _ = op(sl, value, RANDOM.randrange(100))

        self.run_pedantic(f)


@cache
def _del_positions(size: int, count: int, width: int) -> Seq[int]:
    return (
        Range(count)
        .iter()
        .map(lambda i: RANDOM.randrange(size - (i + 1) * width + 1))
        .collect(Seq)
    )


def _calls_nb(size: int) -> int:
    return min(size // 4, 1_000)


def _values_sample(size: int) -> Seq[int]:
    return _shuffled(size).iter().take(_calls_nb(size)).collect(Seq)


@cache
def _shuffled(size: int) -> Vec[int]:
    return Vec.wrap(RANDOM.sample(range(size), size))
