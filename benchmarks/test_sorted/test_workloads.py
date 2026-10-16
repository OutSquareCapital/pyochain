from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Iter

from .._utils import SIZE_PARAMS, BenchFixture
from . import methods
from ._utils import CLS_PARAMS, RANDOM, BenchConfig

if TYPE_CHECKING:
    from ._utils import List


@CLS_PARAMS
@SIZE_PARAMS
def test_priority_queue(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    add_method = methods.add(cls)

    def op(sl: List, value: int, pos: int) -> None:
        if pos < 40:
            add_method(sl, value)
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
    add_method = methods.add(cls)

    def op(sl: List, value: int, pos: int) -> None:
        if pos < 75:
            assert value in sl
        elif pos < 85:
            add_method(sl, value)
        elif pos < 95:
            sl.remove(value)
        elif value < len(sl):
            _ = sl[value]

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_ranking(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    add_method = methods.add(cls)

    def op(sl: List, value: int, pos: int) -> None:
        if pos < 40:
            if value < len(sl):
                _ = sl[value]
        elif pos < 80:
            assert sl.index(value) >= 0
        elif pos < 90:
            add_method(sl, value)
        else:
            sl.remove(value)

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_neighbor(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    add_method = methods.add(cls)

    def op(sl: List, value: int, pos: int) -> None:
        if pos < 75:
            _ = sl.bisect_right(value)
        elif pos < 85:
            add_method(sl, value)
        elif pos < 95:
            sl.remove(value)
        else:
            _ = Iter(sl).take(102).last()

    BenchConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SIZE_PARAMS
def test_intervals(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    config = BenchConfig(benchmark, cls, size)
    add_method = methods.add(cls)

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
            add_method(sl, value)
        else:
            sl.discard(value)

    config.run_workload(op)
