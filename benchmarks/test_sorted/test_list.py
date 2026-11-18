from __future__ import annotations

import pytest

from pyochain import Seq
from pyochain.collections import SortedList

from .._utils import SIZE_PARAMS, BenchFixture
from ._utils import (
    LST_PARAMS,
    RANDOM,
    SC_SIZES,
    SEED,
    List,
    add_fn,
    run_pedantic,
    shuffled,
)
from ._workloads import WORKLOADS, WorkLoad


@pytest.mark.parametrize(
    "step", (pytest.param(1, id="large"), pytest.param(10, id="small"))
)
@LST_PARAMS
@SIZE_PARAMS
def test_update(benchmark: BenchFixture, cls: type[List], size: int, step: int) -> None:
    values = shuffled(size, step)
    if cls is SortedList:
        run_pedantic(benchmark, SortedList, size, lambda lst: lst.extend(values))
    else:
        run_pedantic(benchmark, cls, size, lambda lst: lst.update(values))


@pytest.mark.parametrize("op", WORKLOADS)
@LST_PARAMS
@SC_SIZES
def test_workload(
    benchmark: BenchFixture, cls: type[List], size: int, op: WorkLoad
) -> None:
    add_method = add_fn(cls)
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

    run_pedantic(benchmark, cls, size, f)
