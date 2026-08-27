from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

from ._utils import SIZES, Sizes

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_2(benchmark: BenchFixture, size: int) -> None:
    data = Range(size).iter().enumerate().collect(Seq)
    assert benchmark(_2, data, size)[0] % 2 == 0


def _2(data: Seq[tuple[int, int]], size: int) -> tuple[int, int]:
    f: Callable[[int, int], bool] = lambda x, _a: x % 2 == 0
    for _ in range(SIZES[size]):
        _ = data.iter().filter_star(f).last()
    return data.iter().filter_star(f).last()


def test_3(benchmark: BenchFixture) -> None:
    data = Range(Sizes.SIZE_4096).iter().map(lambda x: (x, x + 1, x + 2)).collect(Seq)
    assert benchmark(_3, data)[0] % 2 == 0


def _3(data: Seq[tuple[int, int, int]]) -> tuple[int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b: x % 2 == 0).last()


def test_4(benchmark: BenchFixture) -> None:
    data = (
        Range(Sizes.SIZE_4096)
        .iter()
        .map(lambda x: (x, x + 1, x + 2, x + 3))
        .collect(Seq)
    )
    assert benchmark(_4, data)[0] % 2 == 0


def _4(data: Seq[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b, _c: x % 2 == 0).last()


def test_5(benchmark: BenchFixture) -> None:
    data = (
        Range(Sizes.SIZE_4096)
        .iter()
        .map(lambda x: (x, x + 1, x + 2, x + 3, x + 4))
        .collect(Seq)
    )
    assert benchmark(_5, data)[0] % 2 == 0


def _5(data: Seq[tuple[int, int, int, int, int]]) -> tuple[int, int, int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b, _c, _d: x % 2 == 0).last()
