from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Range
from pyochain.collections import SortedSet

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "fn", (SortedSet[int].update, SortedSet[int].difference_update)
)
def test_update_set_deadlock(
    fn: Callable[[SortedSet[int], SortedSet[int]], None],
) -> None:
    data = Range(3)
    a = SortedSet(data)
    assert fn(a, a) == a
