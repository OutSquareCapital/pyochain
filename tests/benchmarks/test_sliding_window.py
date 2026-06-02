from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group="map_windows")
@pytest.mark.parametrize("size", [2, 8, 32, 128])
def test_map_windows(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, 4096)
    assert benchmark(_map_windows, data, size).last() is not None


def _map_windows(data: Range, size: int) -> Seq[tuple[int, ...]]:
    return data.iter().map_windows(size, lambda x: x).collect(Seq)
