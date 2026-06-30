from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Vec

from ._utils import SIZES, Sizes

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_retain(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_retain, data.pipe(Vec)) is None
    canary = data.pipe(Vec)
    canary.retain(lambda i: i < 5)
    assert canary.inner == [0, 1, 2, 3, 4]


def _retain(data: Vec[int]) -> None:
    return data.retain(lambda i: i % 2 == 0)


def test_from_list(benchmark: BenchFixture) -> None:
    data = [1, 2, 3]
    assert benchmark(Vec, data) is not None
    canary = Vec(data)
    assert canary.inner == [1, 2, 3]


def test_truncate(benchmark: BenchFixture) -> None:
    data = Range(0, Sizes.SIZE_4096)

    def fn() -> None:
        return data.pipe(Vec).truncate(1)

    assert benchmark(fn) is None
    v = data.pipe(Vec)
    v.truncate(1)
    assert v.first() == 0


def test_drain(benchmark: BenchFixture) -> None:
    data = Range(0, Sizes.SIZE_4096)

    def fn() -> int:
        return data.pipe(Vec).drain().count()

    assert benchmark(fn) == Sizes.SIZE_4096


def test_extract_if(benchmark: BenchFixture) -> None:
    data = Range(0, Sizes.SIZE_4096)

    def fn() -> int:
        return data.pipe(Vec).extract_if(lambda _: True).last()

    assert benchmark(fn) == Sizes.SIZE_4096 - 1
