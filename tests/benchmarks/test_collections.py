from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range, Set

from ._utils import SIZES

if TYPE_CHECKING:
    from pyochain.abc import PyoContainer

    from ._utils import BenchFixture


@pytest.mark.parametrize("size", SIZES)
def test_contains(benchmark: BenchFixture, size: int) -> None:
    data = Range(0, size).pipe(Set)
    assert benchmark(_contains, data, size) is True


def _contains(data: PyoContainer[int], size: int) -> bool:
    val = size - 1
    for _ in range(size):
        _ = data.contains(val)
    return data.contains(val)
