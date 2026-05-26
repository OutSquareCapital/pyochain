from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range

from ._utils import SIZES

if TYPE_CHECKING:
    from ._utils import BenchFixture


def _last(data: Range) -> int:
    return data.iter().last()


def _count(data: Range) -> int:
    return data.iter().count()


@pytest.mark.benchmark(group="last")
@pytest.mark.parametrize("size", SIZES)
def test_last(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_last, data) == size - 1


@pytest.mark.benchmark(group="count")
@pytest.mark.parametrize("size", SIZES)
def test_count(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_count, data) == size
