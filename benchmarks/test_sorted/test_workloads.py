from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pyochain import Iter, Seq

from .._utils import BenchFixture, consume
from . import methods
from ._utils import CLS_PARAMS, RANDOM, SC_SIZES, SEED, SeqConfig, shuffled

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _pytest.mark import ParameterSet

    from ._utils import List

type AddFn[S] = Callable[[S, int], None]


def workloads() -> Seq[ParameterSet]:
    return (
        Iter(_priority_queue, _multiset, _ranking, _neighbor, _intervals)
        .map(lambda fn: pytest.param(fn, id=fn.__name__.removeprefix("_")))
        .collect(Seq)
    )


@CLS_PARAMS
@SC_SIZES
@pytest.mark.parametrize("op", workloads())
def test_workload(
    benchmark: BenchFixture,
    cls: type[List],
    size: int,
    op: Callable[[List, int, int, AddFn[List]], object],
) -> None:
    add_method = methods.add(cls)
    values = (
        shuffled(size, 10)
        .tap(lambda _: RANDOM.seed(SEED))
        .iter()
        .map(lambda val: (val, RANDOM.randrange(start=100)))
        .collect(Seq)
    )

    def f(sl: List) -> None:
        for value, pos in values:
            _ = op(sl, value, pos, add_method)

    SeqConfig(benchmark, cls, size).run_pedantic(f, filled=True)


def _priority_queue(sl: List, value: int, pos: int, add_method: AddFn[List]) -> None:
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


def _multiset(sl: List, value: int, pos: int, add_method: AddFn[List]) -> None:
    if pos < 75:
        assert value in sl
    elif pos < 85:
        add_method(sl, value)
    elif pos < 95:
        sl.remove(value)
    elif value < len(sl):
        _ = sl[value]


def _ranking(sl: List, value: int, pos: int, add_method: AddFn[List]) -> None:
    if pos < 40:
        if value < len(sl):
            _ = sl[value]
    elif pos < 80:
        assert sl.index(value) >= 0
    elif pos < 90:
        add_method(sl, value)
    else:
        sl.remove(value)


def _neighbor(sl: List, value: int, pos: int, add_method: AddFn[List]) -> None:
    if pos < 75:
        _ = sl.bisect_right(value)
    elif pos < 85:
        add_method(sl, value)
    elif pos < 95:
        sl.remove(value)
    else:
        _iterate(sl)


def _intervals(sl: List, value: int, pos: int, add_method: AddFn[List]) -> None:
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


def _iterate(sl: Iterable[object]) -> None:
    Iter(sl).take(102).pipe(consume)
