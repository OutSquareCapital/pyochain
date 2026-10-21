from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from benchmarks.test_sorted import methods
from pyochain import Seq

from .._utils import SIZE_PARAMS, BenchFixture, consume
from ._utils import (
    CLS_DICT_PARAMS,
    SC_SIZES,
    Dict,
    DictConfig,
    InitDict,
    item_to_pair,
    shuffled,
)
from .methods import OpFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.mark import ParameterSet

OPS: Final[Sequence[ParameterSet]] = OpFactory[Dict].generate(
    OpFactory[Dict]("contains", lambda dct: dct.__contains__),
    OpFactory[Dict]("getitem", lambda dct: dct.__getitem__),
)


@CLS_DICT_PARAMS
@SC_SIZES
@pytest.mark.parametrize(
    "filled", (pytest.param(True, id="filled"), pytest.param(False, id="empty"))
)
def test_setitem(
    benchmark: BenchFixture, cls: type[Dict], size: int, filled: bool
) -> None:

    values = shuffled(size, 100).iter().map(item_to_pair).collect(Seq)

    def f(dct: Dict) -> None:
        values.iter().for_each_star(dct.__setitem__)

    DictConfig(benchmark, cls, size).run_pedantic(f, filled=filled)


@CLS_DICT_PARAMS
@SC_SIZES
def test_delitem(benchmark: BenchFixture, cls: type[Dict], size: int) -> None:
    values = shuffled(size, 100)
    DictConfig(benchmark, cls, size).run_pedantic(methods.delitem(values))


@pytest.mark.parametrize("op", OPS)
@CLS_DICT_PARAMS
@SC_SIZES
def test_op(
    benchmark: BenchFixture, cls: type[Dict], size: int, op: OpFactory[Dict]
) -> None:
    config = DictConfig(benchmark, cls, size)
    values = shuffled(config.size, 100)
    f = op.fn(config.new_from_size())
    config.bench(lambda: values.iter().for_each(f))


@CLS_DICT_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: type[Dict], size: int) -> None:
    dct = DictConfig(benchmark, cls, size).new_from_size()
    benchmark(consume, dct)


@CLS_DICT_PARAMS
@SIZE_PARAMS
def test_init(benchmark: BenchFixture, cls: InitDict[Dict], size: int) -> None:
    data = shuffled(size).iter().map(item_to_pair).collect(tuple)
    assert benchmark(cls, data)
