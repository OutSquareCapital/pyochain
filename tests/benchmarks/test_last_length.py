from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range

if TYPE_CHECKING:
    from ._utils import BenchFixture

SIZES = [100, 500, 2500]


def _last(data: Range) -> int:
    return data.iter().last()


def _length(data: Range) -> int:
    return data.iter().length()


@pytest.mark.benchmark(group="last")
@pytest.mark.parametrize("size", SIZES)
def test_last(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_last, data) == size - 1


@pytest.mark.benchmark(group="length")
@pytest.mark.parametrize("size", SIZES)
def test_length(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_length, data) == size
