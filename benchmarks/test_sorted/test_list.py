from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from pyochain import Range, Seq

from .._utils import SIZE_PARAMS, BenchFixture
from . import methods
from ._utils import CLS_PARAMS, RANDOM, SC_SIZES, SEED, List, SeqConfig, shuffled
from ._workloads import WORKLOADS, WorkLoad
from .methods import OpFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.mark import ParameterSet

OPS: Final[Sequence[ParameterSet]] = OpFactory[List].generate(
    OpFactory[List]("bisect", lambda lst: lst.bisect_right),
    OpFactory[List]("index", lambda lst: lst.index),
    OpFactory[List]("count", lambda lst: lst.count),
    OpFactory[List]("contains", lambda lst: lst.__contains__),
    OpFactory[List]("getitem", lambda lst: lst.__getitem__),
)


@CLS_PARAMS
@SC_SIZES
def test_add[L: List](benchmark: BenchFixture, cls: type[L], size: int) -> None:
    add_method = methods.add(cls)
    values = shuffled(size, 100)

    def f(lst: L) -> None:
        for val in values:
            add_method(lst, val)

    SeqConfig(benchmark, cls, size).run_pedantic(f)


@CLS_PARAMS
@SC_SIZES
def test_delitem(benchmark: BenchFixture, cls: type[List], size: int) -> None:
    RANDOM.seed(SEED)
    f = (
        Range(size // 100)
        .iter()
        .map(lambda val: RANDOM.randrange(size - val))
        .collect(Seq)
        .pipe(methods.delitem)
    )
    SeqConfig(benchmark, cls, size).run_pedantic(f)


@pytest.mark.parametrize(
    "step", (pytest.param(1, id="large"), pytest.param(10, id="small"))
)
@CLS_PARAMS
@SIZE_PARAMS
def test_update(benchmark: BenchFixture, cls: type[List], size: int, step: int) -> None:
    config = SeqConfig(benchmark, cls, size)
    values = shuffled(config.size, step)
    f = methods.update(config.cls)
    config.run_pedantic(lambda lst: f(lst, values))


@pytest.mark.parametrize("op", OPS)
@CLS_PARAMS
@SC_SIZES
def test_op(
    benchmark: BenchFixture, cls: type[List], size: int, op: OpFactory[List]
) -> None:
    config = SeqConfig(benchmark, cls, size)
    values = shuffled(config.size, 100)
    f = op.fn(config.new_from_size())
    config.bench(lambda: values.iter().for_each(f))


@CLS_PARAMS
@SC_SIZES
@pytest.mark.parametrize("op", WORKLOADS)
def test_workload(
    benchmark: BenchFixture, cls: type[List], size: int, op: WorkLoad
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
