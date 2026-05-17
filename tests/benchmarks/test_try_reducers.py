from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Ok, Range

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group="try_find")
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_find(benchmark: BenchFixture, size: int) -> None:
    unreachable = size + 1

    def fn(data: Range) -> None:
        _ = data.iter().try_find(lambda value: Ok(value == unreachable))

    data = Range(0, size)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="try_fold")
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_fold(benchmark: BenchFixture, size: int) -> None:
    def fn(data: Range) -> None:
        _ = data.iter().try_fold(0, lambda accumulator, value: Ok(accumulator + value))

    data = Range(0, size)
    assert benchmark(fn, data) is None


@pytest.mark.benchmark(group="try_reduce")
@pytest.mark.parametrize("size", [10, 100, 500])
def test_try_reduce(benchmark: BenchFixture, size: int) -> None:
    def fn(data: Range) -> None:
        _ = data.iter().try_reduce(lambda left, right: Ok(left + right))

    data = Range(0, size)
    assert benchmark(fn, data) is None
