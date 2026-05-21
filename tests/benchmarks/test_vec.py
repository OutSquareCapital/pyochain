from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Vec

from ._utils import SIZES

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_retain(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_retain, data.into(Vec)) is None
    canary = data.into(Vec)
    canary.retain(lambda i: i < 5)
    assert canary.inner == [0, 1, 2, 3, 4]


def _retain(data: Vec[int]) -> None:
    return data.retain(lambda i: i % 2 == 0)


def test_from_ref(benchmark: BenchFixture) -> None:
    data = [1, 2, 3]
    assert benchmark(Vec.from_ref, data) is not None
    canary = Vec.from_ref(data)
    assert canary.inner == [1, 2, 3]
