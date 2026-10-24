from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import pytest

from pyochain import Iter, Range, Seq

from ._utils import SC_SIZES, SET_PARAMS, Set, run_pedantic

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import MarkDecorator

    from .._utils import BenchFixture


class _ParamEnum(Enum):
    @classmethod
    def as_params(cls) -> MarkDecorator:
        return (
            Iter(cls)
            .map(lambda slf: pytest.param(slf, id=slf.name))
            .collect(Seq)
            .pipe(lambda params: pytest.mark.parametrize(cls.__name__.lower(), params))
        )


class Op(_ParamEnum):
    Union = auto()
    Difference = auto()
    Intersection = auto()
    SymmetricDifference = auto()

    def get(self, size: int, stop: Stop, update: bool) -> Callable[[Set], object]:  # ruff: ignore[too-many-return-statements]
        start = size if self is self.Union else size // 2
        r = range(start, start + stop.get(size))
        match self, update:
            case self.Union, False:
                return lambda lst: lst.union(r)
            case self.Difference, False:
                return lambda lst: lst.difference(r)
            case self.Intersection, False:
                return lambda lst: lst.intersection(r)
            case self.SymmetricDifference, False:
                return lambda lst: lst.symmetric_difference(r)
            case self.Union, True:
                return lambda lst: lst.update(r)
            case self.Difference, True:
                return lambda lst: lst.difference_update(r)
            case self.Intersection, True:
                return lambda lst: lst.intersection_update(r)
            case self.SymmetricDifference, True:
                return lambda lst: lst.symmetric_difference_update(r)


class Stop(_ParamEnum):
    Tiny = auto()
    Small = auto()
    Medium = auto()
    Large = auto()

    def get(self, size: int) -> int:
        match self:
            case self.Tiny:
                return min(10, size)
            case self.Small:
                return size // 10
            case self.Medium:
                return size * 5 // 10
            case self.Large:
                return size * 9 // 10


@Op.as_params()
@Stop.as_params()
@pytest.mark.parametrize("update", (True, False))
@SET_PARAMS
@SC_SIZES
def test_op(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    benchmark: BenchFixture, cls: type[Set], size: int, stop: Stop, op: Op, update: bool
) -> None:
    f = op.get(size, stop, update)
    if update:
        run_pedantic(benchmark, cls, size, f)
    else:
        _ = benchmark(f, cls(Range(size)))
