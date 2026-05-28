from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._utils import BenchFixture
SIZE = 4096


def _run[T](data: Seq[T]) -> T:
    return data.iter().unique().last()


def _run_by[T, U](data: Seq[T], key: Callable[[T], U]) -> T:
    return data.iter().unique_by(key).last()


@pytest.mark.benchmark(group="unique")
def test_unique(benchmark: BenchFixture) -> None:
    def _fn(x: int) -> int:
        return 0 if x == SIZE - 1 else x

    data = Range(0, SIZE).iter().map(_fn).collect()
    assert benchmark(_run, data) == SIZE - 2


@pytest.mark.benchmark(group="unique_by")
def test_unique_by(benchmark: BenchFixture) -> None:
    def _fn(x: int) -> int:
        return 0 if x == SIZE - 1 else x

    data = Range(0, SIZE).iter().map(_fn).collect()
    assert benchmark(_run_by, data, lambda x: x + 10 - 10) == SIZE - 2
