from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarks._utils import SIZE_PARAMS
from pyochain import Range

if TYPE_CHECKING:
    from ._utils import BenchFixture


def _last(data: Range) -> int:
    return data.iter().last()


def _count(data: Range) -> int:
    return data.iter().count()


@pytest.mark.benchmark(group="last")
@SIZE_PARAMS
def test_last(benchmark: BenchFixture, size: int) -> None:
    data = Range(size)
    assert benchmark(_last, data) == size - 1


@pytest.mark.benchmark(group="count")
@SIZE_PARAMS
def test_count(benchmark: BenchFixture, size: int) -> None:
    data = Range(size)
    assert benchmark(_count, data) == size
