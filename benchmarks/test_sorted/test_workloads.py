from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Iter

from .._utils import BenchFixture, consume
from . import methods
from ._utils import CLS_PARAMS, RANDOM, SC_SIZES, SeqConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._utils import List


@CLS_PARAMS
@SC_SIZES
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
            _iterate(sl)

    SeqConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SC_SIZES
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

    SeqConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SC_SIZES
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

    SeqConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SC_SIZES
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
            _iterate(sl)

    SeqConfig(benchmark, cls, size).run_workload(op)


@CLS_PARAMS
@SC_SIZES
def test_intervals(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    config = SeqConfig(benchmark, cls, size)
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
            list_length = len(sl)
            if value < list_length:
                stop = min(value + 100, list_length)
                other = RANDOM.randrange(value, stop)
                _ = sl[value:other]
        elif pos < 90:
            add_method(sl, value)
        else:
            sl.discard(value)

    config.run_workload(op)


def _iterate(sl: Iterable[object]) -> None:
    Iter(sl).take(102).pipe(consume)
