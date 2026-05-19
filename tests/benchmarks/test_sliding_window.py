from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Seq

from ._utils import SIZES

if TYPE_CHECKING:
    from ._utils import BenchFixture


@pytest.mark.benchmark(group="map_windows")
@pytest.mark.parametrize("size", SIZES)
def test_map_windows(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_map_windows, data).last() is not None


def _map_windows(data: Range) -> Seq[tuple[int, ...]]:
    return data.iter().map_windows(20, lambda x: x).collect()
