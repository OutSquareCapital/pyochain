from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Final

import pytest

from pyochain import Iter, Seq

from ._utils import CLS_SET_PARAMS, SC_SIZES, SeqConfig, Set, Stop, shuffled
from .methods import SetOp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.mark import ParameterSet

    from .._utils import BenchFixture

STOP_PARAMS = pytest.mark.parametrize(
    "stop",
    Iter(Stop).map(lambda op: pytest.param(op, id=op.name)).collect(tuple),
)

OPS: Final[Sequence[ParameterSet]] = SetOp.generate(
    SetOp("union", lambda lst, r: lst.union(r), False),
    SetOp("difference", lambda lst, r: lst.difference(r)),
    SetOp("intersection", lambda lst, r: lst.intersection(r)),
    SetOp("symmetric_difference", lambda lst, r: lst.symmetric_difference(r)),
    SetOp("update", lambda lst, r: lst.update(r), False, True),
    SetOp("difference_update", lambda lst, r: lst.difference_update(r), mut=True),
    SetOp("intersection_update", lambda lst, r: lst.intersection_update(r), mut=True),
    SetOp(
        "symmetric_difference_update",
        lambda lst, r: lst.symmetric_difference_update(r),
        mut=True,
    ),
)


@CLS_SET_PARAMS
@SC_SIZES
def test_contains(benchmark: BenchFixture, cls: type[Set], size: int) -> None:
    config = SeqConfig(benchmark, cls, size)
    values = shuffled(config.size, 100)
    lst = config.new_from_size()
    config.bench(lambda: values.iter().for_each(lst.__contains__))


@CLS_SET_PARAMS
@SC_SIZES
def test_add(benchmark: BenchFixture, cls: type[Set], size: int) -> None:
    values = shuffled(size, 100).iter().map(operator.neg).collect(Seq)

    def f(lst: Set) -> None:
        values.iter().for_each(lst.add)

    SeqConfig(benchmark, cls, size).run_pedantic(f)


@pytest.mark.parametrize("op", OPS)
@STOP_PARAMS
@CLS_SET_PARAMS
@SC_SIZES
def test_op(
    benchmark: BenchFixture, cls: type[Set], size: int, stop: Stop, op: SetOp
) -> None:
    config = SeqConfig(benchmark, cls, size)
    start = config.size // 2 if op.halved else config.size
    r = range(start, start + stop.fn(config.size))
    if op.mut:
        config.run_pedantic(lambda lst: op.fn(lst, r))
    else:
        lst = config.new_from_size()
        _ = config.bench(lambda: op.fn(lst, r))
