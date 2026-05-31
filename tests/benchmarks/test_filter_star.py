from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

from ._utils import Sizes

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group="filter_star")
def test_filter_star_2tuple(benchmark: BenchFixture) -> None:
    data = Range(0, Sizes.SIZE_4096).iter().map(lambda x: (x, x + 1)).collect()
    assert benchmark(_filter_star_2tuple, data)[0] % 2 == 0


def _filter_star_2tuple(data: Seq[tuple[int, int]]) -> tuple[int, int]:
    return data.iter().filter_star(lambda x, _a: x % 2 == 0).last()


@pytest.mark.benchmark(group="filter_star")
def test_filter_star_3tuple(benchmark: BenchFixture) -> None:
    data = Range(0, Sizes.SIZE_4096).iter().map(lambda x: (x, x + 1, x + 2)).collect()
    assert benchmark(_filter_star_3tuple, data)[0] % 2 == 0


def _filter_star_3tuple(data: Seq[tuple[int, int, int]]) -> tuple[int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b: x % 2 == 0).last()


@pytest.mark.benchmark(group="filter_star")
def test_filter_star_4tuple(benchmark: BenchFixture) -> None:
    data = (
        Range(0, Sizes.SIZE_4096)
        .iter()
        .map(lambda x: (x, x + 1, x + 2, x + 3))
        .collect()
    )
    assert benchmark(_filter_star_4tuple, data)[0] % 2 == 0


def _filter_star_4tuple(
    data: Seq[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b, _c: x % 2 == 0).last()


@pytest.mark.benchmark(group="filter_star")
def test_filter_star_5tuple(benchmark: BenchFixture) -> None:
    data = (
        Range(0, Sizes.SIZE_4096)
        .iter()
        .map(lambda x: (x, x + 1, x + 2, x + 3, x + 4))
        .collect()
    )
    assert benchmark(_filter_star_5tuple, data)[0] % 2 == 0


def _filter_star_5tuple(
    data: Seq[tuple[int, int, int, int, int]],
) -> tuple[int, int, int, int, int]:
    return data.iter().filter_star(lambda x, _a, _b, _c, _d: x % 2 == 0).last()
