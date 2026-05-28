from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

from ._utils import SIZES

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._utils import BenchFixture


def _run[T](data: Seq[T]) -> bool:
    return data.iter().all_unique()


def _run_by[T, U](data: Seq[T], key: Callable[[T], U]) -> bool:
    return data.iter().all_unique_by(key)


@pytest.mark.benchmark(group="all_unique")
@pytest.mark.parametrize("size", SIZES)
def test_int(benchmark: BenchFixture, size: int) -> None:
    def _fn(x: int) -> int:
        return 0 if x == size - 1 else x

    data = Range(0, size).iter().map(_fn).collect()
    assert benchmark(_run, data) is False


@pytest.mark.benchmark(group="all_unique")
@pytest.mark.parametrize("size", SIZES)
def test_str(benchmark: BenchFixture, size: int) -> None:
    def _fn(i: int) -> str:
        return "0" if i == size - 1 else str(i)

    data = Range(0, size).iter().map(_fn).collect()
    assert benchmark(_run, data) is False


@pytest.mark.benchmark(group="all_unique_by")
@pytest.mark.parametrize("size", SIZES)
def test_int_by(benchmark: BenchFixture, size: int) -> None:
    def _fn(x: int) -> int:
        return 0 if x == size - 1 else x

    data = Range(0, size).iter().map(_fn).collect()
    assert benchmark(_run_by, data, lambda x: x + 10 - 10) is False


@pytest.mark.benchmark(group="all_unique_by")
@pytest.mark.parametrize("size", SIZES)
def test_str_by(benchmark: BenchFixture, size: int) -> None:
    def _fn(i: int) -> str:
        return "0" if i == size - 1 else str(i)

    data = Range(0, size).iter().map(_fn).collect()
    assert benchmark(_run_by, data, str.lower) is False
