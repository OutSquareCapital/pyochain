from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range

from ._utils import SIZES

if TYPE_CHECKING:
    from pyochain.abc import PyoReversible

    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_rev(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size)
    assert benchmark(_rev, data, size) == 1


def _rev(data: PyoReversible[int], size: int) -> int:
    for _ in range(size):
        _ = data.rev()

    return 1
